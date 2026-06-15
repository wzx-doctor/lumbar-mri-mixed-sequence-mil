# -*- coding: utf-8 -*-
"""
train_utils.py  —  ABMIL 训练与评估工具（科研风格绘图，单轴对比）

功能列表：
1) 训练与验证：train_one_epoch / validate / fit
2) 早停与模型保存：EarlyStopping（按 val_macro_auc 或 val_loss）
3) 阈值选择：Youden / MaxF1；多标签逐类选择阈值
4) 指标计算：ROC-AUC、PR-AP、F1/P/R、混淆矩阵、分类报告
5) 预测导出：保存 y_true / y_score / y_pred / 阈值
6) 绘图（单轴对比，科研风格，300dpi）：
   - ROC（各类 + micro 同轴）
   - PR（各类 + micro 同轴，含 prevalence 基线）
   - Calibration（各类折线 + 理想虚线）
   - Decision Curve Analysis（各类 + treat-none/treat-all）
   - F1 / AUC 柱状图
   - 训练-验证曲线（单图双 y 轴：loss 与 val macro AUC），文件名：train_val_loss_auc.png
7) 注意力可视化（ABMIL）：Top-k 切片拼图 + 注意力折线 + CSV
   - 反归一化：默认 z-score（normalize_mode="zscore"）
   - 分位裁剪 + gamma 矫正 + 柔和叠加，避免过亮

依赖：numpy, pandas, matplotlib, scikit-learn, torch
"""
from __future__ import annotations
import os
import json
import math
import random
import warnings
import shutil
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch import amp as torch_amp
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
from typing import Optional, Dict, Any
# --- loss builder: BCE(pos_weight) / Focal(BCE) 二合一 ---
import torch.nn.functional as F
import csv
from pathlib import Path
# ---- strong prior logging switch ----
PRIOR_VERBOSE = False           # 控制是否打印逐样本 MISS/MATCH（默认关闭）
PRIOR_PRINT_EVERY = 200         # 命中计数到这个间隔才轻量打印一次

class _FocalWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.pos_weight = pos_weight  # shape [C] or None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # per-element BCE with logits
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0, p, 1.0 - p)   # p_t
        alpha = torch.where(targets > 0, self.alpha, 1.0 - self.alpha)
        loss = alpha * (1 - pt).pow(self.gamma) * bce
        return loss.mean()
        
def build_loss(pos_weight: Optional[torch.Tensor] = None,
               focal_gamma: Optional[float] = None,
               focal_alpha: float = 0.25):
    if focal_gamma is None:                          # 纯 BCE
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:                                            # Focal(BCE)
        return _FocalWithLogits(gamma=focal_gamma, alpha=focal_alpha, pos_weight=pos_weight)

def plot_train_val_loss_auc(
    train_losses: List[float],
    val_losses: List[float],
    val_macro_auc: List[float],
    out_path: str,
    title_prefix: str = ""
) -> None:
    """
    在一张图里画两个子图：
      左：Train & Val Loss
      右：Validation Macro AUC
    保存为 out_path （如: <out_dir>/train_val_loss_auc.png）
    """
    import matplotlib.pyplot as plt
    import numpy as np

    epochs = np.arange(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=120)

    # 左：Loss
    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses,   label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title_prefix} Train & Val Loss".strip())
    axes[0].legend(frameon=False)

    # 右：Val Macro AUC
    axes[1].plot(epochs, val_macro_auc, label="Val Macro AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title(f"{title_prefix} Validation Macro AUC".strip())
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def organize_outputs(out_dir: str) -> None:
    """
    把 out_dir 下的图片类文件放到 figures/，表格/数组类文件放到 tables/。
    若目标文件已存在：优先覆盖；若覆盖失败则自动改名避免报错。
    """
    import os, shutil

    figs_ext = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
    tbls_ext = {".csv", ".json", ".npz", ".npy"}

    out_dir = str(out_dir)
    figures_dir = os.path.join(out_dir, "figures")
    tables_dir  = os.path.join(out_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir,  exist_ok=True)

    def _safe_move(src_path: str, dst_dir: str) -> None:
        """覆盖式移动；若仍失败则追加 _1,_2… 改名后再移。"""
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        # 已在目标处则跳过
        if os.path.abspath(src_path) == os.path.abspath(dst_path):
            return
        try:
            # 优先覆盖（同一分区下用 os.replace 原子覆盖）
            if os.path.exists(dst_path):
                os.replace(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)
        except Exception:
            # 覆盖失败（例如跨盘或被占用）→ 自动改名
            stem, ext = os.path.splitext(dst_path)
            k = 1
            cand = f"{stem}_{k}{ext}"
            while os.path.exists(cand):
                k += 1
                cand = f"{stem}_{k}{ext}"
            shutil.move(src_path, cand)

    # 遍历 out_dir 顶层文件（不递归），分门别类移动
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in figs_ext:
            _safe_move(path, figures_dir)
        elif ext in tbls_ext:
            _safe_move(path, tables_dir)
        # 其它类型保持在顶层

# ========================= 全局风格与设备 =========================
matplotlib.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 8.0,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
})

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================= 工具函数 =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ========================= 注意力熵工具 =========================
def _entropy_of_w(w: torch.Tensor) -> torch.Tensor:
    """对注意力权重计算熵（越大越均匀）。用于 lambda_entropy * H(w)。"""
    eps = 1e-8
    p = w.clamp_min(eps)
    p = p / p.sum()
    return -(p * p.log()).sum()

# ========================= EarlyStopping =========================
class EarlyStopping:
    """
    监控目标在 patience 个 epoch 内没有改善则早停；保存：
      - best_model_epoch{E}.pth
      - best_model.pth（稳定别名）
      - early_stopping_meta.json
    """

    def __init__(
        self,
        monitor: str = "val_macro_auc",   # "val_macro_auc" 或 "val_loss"
        mode: str = "max",                # 监控方式：max 或 min
        patience: int = 10,
        min_delta: float = 1e-6,
        out_dir: str = "./outputs"
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.out_dir = out_dir

        self.best = None
        self.best_epoch = -1
        self.num_bad = 0
        ensure_dir(out_dir)

    def _is_better(self, score: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return score > (self.best + self.min_delta)
        else:
            return score < (self.best - self.min_delta)

    def step(self, epoch: int, score: float, model: nn.Module) -> bool:
        """
        返回 True 表示需要早停
        """
        if self._is_better(score):
            self.best = score
            self.best_epoch = epoch
            self.num_bad = 0
            # 保存
            torch.save(model.state_dict(), os.path.join(self.out_dir, f"best_model_epoch{epoch}.pth"))
            torch.save(model.state_dict(), os.path.join(self.out_dir, "best_model.pth"))
            meta = {"monitor": self.monitor, "mode": self.mode, "best": float(score), "best_epoch": int(epoch)}
            save_json(os.path.join(self.out_dir, "early_stopping_meta.json"), meta)
            return False
        else:
            self.num_bad += 1
            return self.num_bad >= self.patience


# ========================= 训练/验证 =========================
def train_one_epoch(model, train_loader, criterion, optimizer,
                    scaler=None, lambda_entropy: float = 0.0, grad_clip_norm: float | None = 1.0,
                    lambda_prior: float = 0.0, strong_lookup: dict | None = None):
    model.train()
    device = next(model.parameters()).device
    running_loss = 0.0
    matched_cnt = 0
    posbag_cnt = 0

    def _norm_seq(x: str) -> str:
        return str(x).strip().upper().replace("-", "").replace("_", "")

    def _to_scalar_str(v) -> str:
        if isinstance(v, (list, tuple)):
            v = v[0] if len(v) > 0 else ""
        import torch
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                v = v.item()
            else:
                v = v.detach().cpu().numpy().tolist()
                v = v[0] if isinstance(v, (list, tuple)) and len(v) > 0 else v
        s = str(v).strip()
        if s.startswith("[") and s.endswith("]") and "," not in s:
            s = s.strip("[]").strip().strip("'").strip('"')
        return s

    def unpack_outputs(outputs):
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
            w = outputs[1] if len(outputs) >= 2 else None
        else:
            logits, w = outputs, None
        return logits, w

    def entropy_of_w(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        p = w / (w.sum(dim=0, keepdim=True) + eps)
        p = p.clamp_min(eps)
        return -(p * p.log()).sum()

    for bags, labels, meta in train_loader:
        bags = bags.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # ===== AMP =====
            with torch_amp.autocast("cuda"):
                outputs = model(bags)
                logits, w = unpack_outputs(outputs)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)

                loss = criterion(logits, labels)
                if (w is not None) and (lambda_entropy > 0.0):
                    loss = loss + lambda_entropy * entropy_of_w(w)

                # ---- 强监督注意力先验（只在 AMP 分支内执行一次）----
                if (w is not None) and (lambda_prior > 0.0) and (strong_lookup is not None) and (meta is not None):
                    eps = 1e-6
                    pos_classes = (labels.view(-1) > 0.5).nonzero(as_tuple=True)[0].tolist()
                    if len(pos_classes) > 0:
                        posbag_cnt += 1
                        chk = _to_scalar_str(meta.get("checkno", ""))
                        seq = _norm_seq(_to_scalar_str(meta.get("seq_type", "")))
                        idx_list = torch.as_tensor(meta["slice_idx"]).detach().view(-1).cpu().tolist()
                        w1d = w.reshape(-1)
                        L_prior = 0.0

                        for si in idx_list[:5]:
                            key_with_seq  = (chk,  seq, int(si))
                            key_no_seq    = (chk, int(si))
                            if key_with_seq in strong_lookup:
                                if PRIOR_VERBOSE and (matched_cnt % PRIOR_PRINT_EVERY == 0):
                                    print("[DEBUG] MATCH with seq:", key_with_seq)
                                break
                            elif key_no_seq in strong_lookup:
                                if PRIOR_VERBOSE and (matched_cnt % PRIOR_PRINT_EVERY == 0):
                                    print("[DEBUG] MATCH no seq:", key_no_seq)
                                break
                            else:
                                if PRIOR_VERBOSE and (matched_cnt % PRIOR_PRINT_EVERY == 0):
                                    print("[DEBUG] MISS:", key_with_seq, key_no_seq)


                        for c in pos_classes:
                            rc = []
                            for si in idx_list:
                                key = (chk, seq, int(si)) if len(seq) > 0 else (chk, int(si))
                                vec = strong_lookup.get(key)
                                rc.append(float(vec[c]) if (vec is not None and len(vec) > c) else 0.0)
                            if any(v > 0 for v in rc):
                                matched_cnt += 1
                            r = torch.tensor(rc, device=w.device, dtype=w.dtype)
                            if r.sum() > 0:
                                mass = (w1d * r).sum() / (r.sum() + eps)
                                L_prior = L_prior - torch.log(mass + eps)
                            if L_prior != 0.0 and (matched_cnt % PRIOR_PRINT_EVERY == 0):
                                print(f"[PRIOR] batch L_prior={float(L_prior):.4f} matched_cnt={matched_cnt}")
                        if L_prior != 0:
                            loss = loss + lambda_prior * L_prior
                            if (matched_cnt % 200) == 0:
                                    print(f"[PRIOR] batch L_prior={float(L_prior):.4f} matched_cnt={matched_cnt}")
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        else:
            # ===== 非 AMP =====
            outputs = model(bags)
            logits, w = unpack_outputs(outputs)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            loss = criterion(logits, labels)
            if (w is not None) and (lambda_entropy > 0.0):
                loss = loss + lambda_entropy * entropy_of_w(w)

            # ---- 强监督注意力先验（只在非 AMP 分支内执行一次）----
            if (w is not None) and (lambda_prior > 0.0) and (strong_lookup is not None) and (meta is not None):
                eps = 1e-6
                pos_classes = (labels.view(-1) > 0.5).nonzero(as_tuple=True)[0].detach().cpu().tolist()
                if len(pos_classes) > 0:
                    posbag_cnt += 1
                    chk = _to_scalar_str(meta.get("checkno", ""))
                    seq = _norm_seq(_to_scalar_str(meta.get("seq_type", "")))
                    idx_list = torch.as_tensor(meta["slice_idx"]).detach().view(-1).cpu().tolist()
                    w1d = w.reshape(-1)
                    L_prior = 0.0

                    for si in idx_list[:5]:
                        key_with_seq = (chk, seq, int(si))
                        key_no_seq   = (chk, int(si))
                        if key_with_seq in strong_lookup:
                            if PRIOR_VERBOSE and (matched_cnt % PRIOR_PRINT_EVERY == 0):
                                print("[DEBUG] MATCH with seq:", key_with_seq)
                            break
                        elif key_no_seq in strong_lookup:
                            if PRIOR_VERBOSE and (matched_cnt % PRIOR_PRINT_EVERY == 0):
                                print("[DEBUG] MATCH no seq:", key_no_seq)
                            break
                        else:
                            if PRIOR_VERBOSE and (matched_cnt % PRIOR_PRINT_EVERY == 0):
                                print("[DEBUG] MISS:", key_with_seq, key_no_seq)

                    for c in pos_classes:
                        rc = []
                        for si in idx_list:
                            key = (chk, seq, int(si)) if len(seq) > 0 else (chk, int(si))
                            vec = strong_lookup.get(key)
                            rc.append(float(vec[c]) if (vec is not None and len(vec) > c) else 0.0)
                        if any(v > 0 for v in rc):
                            matched_cnt += 1
                        r = torch.tensor(rc, device=w.device, dtype=w.dtype)
                        if r.sum() > 0:
                            mass = (w1d * r).sum() / (r.sum() + eps)
                            L_prior = L_prior - torch.log(mass + eps)
                        if L_prior != 0.0 and (matched_cnt % PRIOR_PRINT_EVERY == 0):
                            print(f"[PRIOR] batch L_prior={float(L_prior):.4f} matched_cnt={matched_cnt}")
                    if L_prior != 0.0:
                        loss = loss + lambda_prior * L_prior
                        if (matched_cnt % 200) == 0:
                                print(f"[PRIOR] batch L_prior={float(L_prior):.4f} matched_cnt={matched_cnt}")

            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        running_loss += float(loss.item())

    return running_loss / max(1, len(train_loader)), matched_cnt, posbag_cnt

@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    scaler: Optional[torch_amp.GradScaler] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    验证阶段：
      - 与 train_one_epoch 同形状处理：DataLoader 返回 (bags, labels)
      - 去除 batch 维：bags.shape [B, m, 1, H, W] -> [m, 1, H, W]，labels [B, C] -> [C]
      - AMP 分支采用 torch.amp.autocast('cuda', enabled=True)
      - 为避免 half/float 冲突，输入在前向前统一 .float()
    返回：
      - 平均损失（float）
      - y_true: [N, C]（0/1）
      - y_score: [N, C]（sigmoid 概率）
    """
    device = next(model.parameters()).device
    model.eval()

    running = 0.0
    n_batches = 0

    y_true_list: List[np.ndarray] = []
    y_score_list: List[np.ndarray] = []

    pbar = tqdm(loader, desc="valid", leave=False)

    # 修改后：
    for batch in pbar:
        # 兼容两种返回：(bags, labels) 或 (bags, labels, meta)
        if len(batch) == 3:
            bags, labels, meta = batch
        else:
            bags, labels = batch
            meta = None

        # 移动设备并去掉 batch 维
        bags = bags.to(device).squeeze(0)           # [m, 1, H, W]
        labels = labels.to(device).float().squeeze(0)  # [C]

        # AMP 分支（与训练一致）；输入统一为 fp32，避免 conv 权重/偏置与 half 冲突
        if scaler is not None and torch.cuda.is_available():
            with torch_amp.autocast('cuda', enabled=True):
                logits = model(bags.float())
                if isinstance(logits, tuple):
                    logits = logits[0]              # 兼容返回 (logits, attn)
                loss = criterion(logits, labels)
        else:
            logits = model(bags.float())
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, labels)

        running += float(loss.item())
        n_batches += 1

        # 收集 y_true / y_score
        y_true_list.append(labels.detach().cpu().numpy())                  # [C]
        y_score_list.append(torch.sigmoid(logits).detach().cpu().numpy()) # [C]

    avg_loss = running / max(n_batches, 1)

    # 堆叠为 [N, C]
    y_true = np.stack(y_true_list, axis=0)
    y_score = np.stack(y_score_list, axis=0)

    return avg_loss, y_true, y_score

def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs: int,
    out_dir: str,
    monitor: str = "val_macro_auc",        # "val_macro_auc" 或 "val_loss"
    earlystop_patience: int = 10,
    earlystop_delta: float = 0.0,
    use_amp: bool = True,
    device: Optional[torch.device] = None,
    model_name: str = "",
    fold_idx: Optional[int] = None,
    lambda_entropy: float = 0.0,           # ← 已有
    lambda_prior: float = 0.0,             # ← 新增
    strong_lookup: Optional[dict] = None,  # ← 新增
) -> Dict[str, Any]:
    """
    统一训练循环：
      - 支持 AMP（torch.amp）
      - 早停（monitor = 'val_macro_auc' 越大越好；'val_loss' 越小越好）
      - 结束后绘制合并图：train/val loss + val macro AUC 到 train_val_loss_auc.png
      - 返回 history 便于上层复用
    """
    # ---------------- 设备与 AMP ----------------
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)
    scaler = torch_amp.GradScaler("cuda") if (use_amp and torch.cuda.is_available()) else None
    # ---- LR Scheduler: warmup + cosine ----
    warmup_epochs = 5
    total_epochs = num_epochs

    def _lr_lambda(e):
        if e < warmup_epochs:
           return float(e + 1) / max(1, warmup_epochs)
        t = (e - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    min_epochs = 15  # 最少训练轮数，下限保护
    grad_clip_norm = 1.0  # 梯度裁剪范数，供 5B 使用

    # ---------------- 记录容器 ----------------
    train_losses: List[float] = []
    val_losses: List[float] = []
    val_macro_auc: List[float] = []

    best_score = -float("inf") if monitor.lower() != "val_loss" else float("inf")
    best_state_dict = None
    patience_cnt = 0

    ensure_dir(out_dir)

    # ---------------- 训练循环 ----------------
    for epoch in range(1, num_epochs + 1):
        # 1) 训练一个 epoch
        tr_loss, matched_cnt, posbag_cnt = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            lambda_entropy=lambda_entropy,
            lambda_prior=lambda_prior,
            strong_lookup=strong_lookup
    )

        train_losses.append(tr_loss)

        # 2) 验证一个 epoch（拿到 y_true/y_score 用于 AUC）
        val_loss, y_true, y_score = validate(model, val_loader, criterion, scaler)
        val_losses.append(val_loss)

        # 3) 计算 macro AUC（多标签/多类别情况按列算 AUC 后再平均）
        try:
            c = y_true.shape[1]
            auc_list = []
            for ci in range(c):
                # 若某列只有一个类别，会抛异常，跳过
                if len(np.unique(y_true[:, ci])) < 2:
                    continue
                auc_ci = roc_auc_score(y_true[:, ci], y_score[:, ci])
                auc_list.append(auc_ci)
            macro_auc = float(np.mean(auc_list)) if len(auc_list) > 0 else float("nan")
        except Exception:
            macro_auc = float("nan")
        val_macro_auc.append(macro_auc)
        print(f"[PRIOR] epoch={epoch:03d} matched={matched_cnt} / pos_bags={posbag_cnt}")
        # 4) 早停判定
        improved = False
        if monitor.lower() == "val_loss":
            # 越小越好
            if (best_score - val_loss) > earlystop_delta:
                improved = True
                best_score = val_loss
        else:
            # 默认监控 AUC：越大越好
            if (macro_auc - best_score) > earlystop_delta:
                improved = True
                best_score = macro_auc

        if improved:
            patience_cnt = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # === 保存最佳权重与早停元信息（新增） ===
            try:
                os.makedirs(out_dir, exist_ok=True)
        # 带 epoch 的可追溯快照
                torch.save(model.state_dict(), os.path.join(out_dir, f"best_model_epoch{epoch}.pth"))
        # 稳定文件名，供外部直接加载
                torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))

                meta = {
                     "monitor": monitor,          # "val_macro_auc" 或 "val_loss"
                     "best_epoch": int(epoch),
                     "best_score": float(best_score),
                }
                with open(os.path.join(out_dir, "early_stopping_meta.json"), "w", encoding="utf-8") as _f:
                    json.dump(meta, _f, ensure_ascii=False, indent=2)

                print(f"[Best Model] epoch={epoch:03d}  {monitor}={best_score:.4f}")

            except Exception as _e:
                print("[WARN] 保存 early_stopping 元信息失败：", _e)
                
        else:
        # 前 min_epochs 轮不累计耐心
             if epoch < min_epochs:
                 patience_cnt = 0
             else:
                 patience_cnt += 1
        # 5) 进度输出
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_macro_auc={macro_auc:.4f} | best={best_score:.4f} | "
            f"patience={patience_cnt}/{earlystop_patience}"
        )
        print(f"[LR] epoch={epoch:03d} lr={optimizer.param_groups[0]['lr']:.6e}")
        
        scheduler.step()

        # 6) 触发早停
        if (epoch >= min_epochs) and (patience_cnt >= earlystop_patience):
            print(f"[EarlyStop] monitor={monitor} no improvement (delta={earlystop_delta})")
            break

    # ---------------- 训练结束：绘制合并图 ----------------
    # 标题前缀（有则用，无则空）
    title_prefix = ""
    if model_name:
        title_prefix = model_name
    if fold_idx is not None:
        title_prefix = f"{title_prefix}_fold{fold_idx}" if title_prefix else f"fold{fold_idx}"

    plot_train_val_loss_auc(
        train_losses=train_losses,
        val_losses=val_losses,
        val_macro_auc=val_macro_auc,
        out_path=os.path.join(out_dir, "train_val_loss_auc.png"),
        title_prefix=title_prefix,
    )

    # ---------------- 返回历史 ----------------
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_macro_auc": val_macro_auc,
        "best_state_dict": best_state_dict,
        "best_score": best_score,
        "monitor": monitor,
    }
    return history


# ========================= 阈值选择与指标计算 =========================
def select_thresholds_by_youden(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """
    逐类按 Youden 指数（TPR - FPR）选择最优阈值
    """
    C = y_true.shape[1]
    th = np.zeros(C, dtype=np.float32)
    for c in range(C):
        fpr, tpr, thr = roc_curve(y_true[:, c], y_score[:, c])
        j = tpr - fpr
        idx = int(np.argmax(j))
        th[c] = float(thr[idx])
    return th


def select_thresholds_by_maxF1(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """
    逐类在候选阈值上扫描，选择 F1 最大点
    """
    C = y_true.shape[1]
    th = np.zeros(C, dtype=np.float32)
    for c in range(C):
        p, r, t = precision_recall_curve(y_true[:, c], y_score[:, c])
        f1 = 2 * p * r / (p + r + 1e-12)
        idx = int(np.nanargmax(f1))
        # precision_recall_curve 返回的阈值长度比 p/r 少 1，做边界处理
        if idx == 0:
            th[c] = 0.5
        elif idx - 1 < len(t):
            th[c] = float(t[idx - 1])
        else:
            th[c] = float(np.median(t)) if len(t) > 0 else 0.5
    return th
# === 阈值函数别名（保持向后兼容） ===
def select_thresholds_youden(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """别名函数，调用 select_thresholds_by_youden"""
    return select_thresholds_by_youden(y_true, y_score)

def select_thresholds_maxF1(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """别名函数，调用 select_thresholds_by_maxF1"""
    return select_thresholds_by_maxF1(y_true, y_score)


def binarize_with_thresholds(y_score: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (y_score >= thresholds.reshape(1, -1)).astype(np.int32)


def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    C = y_true.shape[1]
    metrics = {
        "f1_per_class": [],
        "precision_per_class": [],
        "recall_per_class": [],
        "macro_f1": None,
        "micro_f1": None
    }
    for c in range(C):
        f1 = f1_score(y_true[:, c], y_pred[:, c], zero_division=0)
        p = precision_score(y_true[:, c], y_pred[:, c], zero_division=0)
        r = recall_score(y_true[:, c], y_pred[:, c], zero_division=0)
        metrics["f1_per_class"].append(float(f1))
        metrics["precision_per_class"].append(float(p))
        metrics["recall_per_class"].append(float(r))
    metrics["macro_f1"] = float(np.mean(metrics["f1_per_class"]))
    metrics["micro_f1"] = float(f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0))
    return metrics

def compute_metrics(y_true: np.ndarray,
                    y_score: np.ndarray,
                    y_pred: np.ndarray,
                    label_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    统一输出键名：
      - Macro_AUC / Micro_AUC / AP_Micro / F1_Macro
      - AUC_per_class 以及 AUC_L3/4, AUC_L4/5, AUC_L5/S1（若类数>=3）
    """
    C = y_true.shape[1]

    # 每类 AUC
    auc_per_class: List[float] = []
    for c in range(C):
        try:
            if len(np.unique(y_true[:, c])) < 2:
                auc_per_class.append(np.nan)
            else:
                auc_per_class.append(roc_auc_score(y_true[:, c], y_score[:, c]))
        except Exception:
            auc_per_class.append(np.nan)

    macro_auc = float(np.nanmean(auc_per_class)) if np.any(~np.isnan(auc_per_class)) else float("nan")
    try:
        micro_auc = float(roc_auc_score(y_true.ravel(), y_score.ravel()))
    except Exception:
        micro_auc = float("nan")
    try:
        ap_micro = float(average_precision_score(y_true.ravel(), y_score.ravel()))
    except Exception:
        ap_micro = float("nan")

    f1_list = [f1_score(y_true[:, c], y_pred[:, c], zero_division=0) for c in range(C)]
    f1_macro = float(np.mean(f1_list))

    out: Dict[str, Any] = {
        "Macro_AUC": macro_auc,
        "Micro_AUC": micro_auc,
        "AP_Micro": ap_micro,
        "F1_Macro": f1_macro,
        "AUC_per_class": [float(x) if not np.isnan(x) else float("nan") for x in auc_per_class],
    }

    # 固定三列名（若 label_cols 提供则按前 3 列映射）
    if label_cols and len(label_cols) >= 3:
        out["AUC_L3/4"] = out["AUC_per_class"][0]
        out["AUC_L4/5"] = out["AUC_per_class"][1]
        out["AUC_L5/S1"] = out["AUC_per_class"][2]
    else:
        if C >= 1: out["AUC_L3/4"] = out["AUC_per_class"][0]
        if C >= 2: out["AUC_L4/5"] = out["AUC_per_class"][1]
        if C >= 3: out["AUC_L5/S1"] = out["AUC_per_class"][2]

    return out

#混淆矩阵
def save_confusion_matrices_png(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], out_dir: str) -> None:
    """
    保存每个类别的混淆矩阵为 PNG 图片
    """
    out_dir = os.path.join(out_dir, "figures", "confusion_matrices")
    ensure_dir(out_dir)

    for c, name in enumerate(class_names):
        cm = confusion_matrix(y_true[:, c], y_pred[:, c], labels=[0, 1])

        fig, ax = plt.subplots(figsize=(3, 3))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)

        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Pred0", "Pred1"],
            yticklabels=["True0", "True1"],
            ylabel="True",
            xlabel="Pred",
            title=f"Confusion Matrix - {name}"
        )

        # 在格子中标注数值
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )

        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, f"confusion_matrix_{name}.png"), dpi=300)
        plt.close(fig)

    for c, name in enumerate(class_names):
        cm = confusion_matrix(y_true[:, c], y_pred[:, c], labels=[0, 1])
        df = pd.DataFrame(cm, index=["True0", "True1"], columns=["Pred0", "Pred1"])
        df.to_csv(os.path.join(out_dir, f"confusion_matrix_{name}.csv"), index=True, encoding="utf-8")


# ========================= 绘图（单轴对比） =========================
def _apply_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
    })


def plot_roc_single_axis(y_true: np.ndarray, y_score: np.ndarray, class_names: List[str], out_path: str, title: str = "ROC Curve") -> None:
    _apply_plot_style()
    C = y_true.shape[1]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for c in range(C):
        fpr, tpr, _ = roc_curve(y_true[:, c], y_score[:, c])
        auc_c = roc_auc_score(y_true[:, c], y_score[:, c])
        ax.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc_c:.2f})", color=colors[c % len(colors)])
    fpr_m, tpr_m, _ = roc_curve(y_true.ravel(), y_score.ravel())
    auc_m = roc_auc_score(y_true, y_score, average="micro")
    ax.plot(fpr_m, tpr_m, "k--", label=f"micro (AUC={auc_m:.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pr_single_axis(y_true: np.ndarray, y_score: np.ndarray, class_names: List[str], out_path: str, title: str = "Precision-Recall Curve") -> None:
    _apply_plot_style()
    C = y_true.shape[1]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for c in range(C):
        p, r, _ = precision_recall_curve(y_true[:, c], y_score[:, c])
        ap = average_precision_score(y_true[:, c], y_score[:, c])
        ax.plot(r, p, label=f"{class_names[c]} (AP={ap:.2f})", color=colors[c % len(colors)])
    p_m, r_m, _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
    ap_m = average_precision_score(y_true, y_score, average="micro")
    ax.plot(r_m, p_m, "k--", label=f"micro (AP={ap_m:.2f})")
    ax.hlines(y_true.mean(), 0, 1, colors="gray", linestyles=":", label=f"baseline={y_true.mean():.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_calibration_single_axis(y_true: np.ndarray, y_score: np.ndarray, class_names: List[str], out_path: str, n_bins: int = 10, title: str = "Calibration Curve") -> None:
    _apply_plot_style()
    C = y_true.shape[1]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for c in range(C):
        prob_true, prob_pred = calibration_curve(y_true[:, c], y_score[:, c], n_bins=n_bins, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label=class_names[c], color=colors[c % len(colors)])
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect Calibration")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_dca_single_axis(y_true: np.ndarray, y_score: np.ndarray, class_names: List[str], out_path: str, title: str = "Decision Curve Analysis") -> None:
    _apply_plot_style()
    thresholds = np.linspace(0.01, 0.99, 100)
    N, C = y_true.shape
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for c in range(C):
        nb = []
        for pt in thresholds:
            pred = (y_score[:, c] >= pt).astype(int)
            TP = np.sum((pred == 1) & (y_true[:, c] == 1))
            FP = np.sum((pred == 1) & (y_true[:, c] == 0))
            nb.append((TP / N) - (FP / N) * (pt / (1 - pt)))
        ax.plot(thresholds, nb, label=class_names[c], color=colors[c % len(colors)])
    ax.plot(thresholds, np.zeros_like(thresholds), "k:", lw=1, label="treat-none")
    prev = y_true.mean()
    ax.plot(thresholds, prev - (1 - prev) * (thresholds / (1 - thresholds)), "k--", lw=1, label="treat-all (micro)")
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_bars_single_axis(values: List[float], class_names: List[str], out_path: str, title: str, ylabel: str) -> None:
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = plt.cm.Blues(np.linspace(0.5, 0.95, len(values)))
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    x = np.arange(len(class_names))
    bars = ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{values[i]:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_train_val_loss_auc_single_axis(train_loss: List[float], val_loss: List[float], val_macro_auc: List[float], out_path: str, title: str) -> None:
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0})
    epochs = np.arange(1, len(train_loss) + 1)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    l1, = ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4")
    l2, = ax1.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    l3, = ax2.plot(epochs, val_macro_auc, label="Val Macro AUC", color="black", linestyle="--")
    ax2.set_ylabel("AUC")
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ========================= 注意力可视化（防过亮） =========================
def _denorm_for_display_zscore(x: np.ndarray, clip_p: Tuple[float, float] = (1.0, 99.0), gamma: float = 1.2) -> np.ndarray:
    """
    假定输入为 z-score 标准化后的切片（均值≈0、方差≈1）。
    执行：逆标准化（用切片内统计近似）→ 分位裁剪 → 线性映射到[0,1] → gamma 矫正（>1 适度压暗）
    """
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x)) if np.std(x) > 1e-6 else 1.0
    x = x * sd + mu
    lo, hi = np.percentile(x, [clip_p[0], clip_p[1]])
    if hi <= lo:
        x01 = np.zeros_like(x, dtype=np.float32)
    else:
        x = np.clip(x, lo, hi)
        x01 = (x - lo) / (hi - lo + 1e-6)
    x01 = np.clip(x01, 0.0, 1.0) ** (1.0 / max(gamma, 1e-6))
    return x01


def _overlay_constant_heat(base01: np.ndarray, w_scalar: float, alpha: float = 0.33, cmap: str = "magma") -> np.ndarray:
    base01 = np.clip(base01, 0.0, 1.0)
    heat01 = np.full_like(base01, float(w_scalar))
    cmap_fn = plt.get_cmap(cmap)
    heat_rgb = cmap_fn(heat01)[..., :3]
    base_rgb = np.stack([base01, base01, base01], axis=-1)
    out = (1.0 - alpha) * base_rgb + alpha * heat_rgb
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


@torch.no_grad()
def test_attention_output(
    model: nn.Module,
    val_loader,
    out_dir: str,
    topk: int = 6,
    normalize_mode: str = "zscore",                     # "zscore" / "none"
    clip_percentile: Tuple[float, float] = (1.0, 99.0), # zscore 可视化截断
    gamma: float = 1.2,                                  # zscore 伽马增强
    alpha: float = 0.33,                                 # 叠加热图透明度
    cmap: str = "magma",                                 # 叠加热图 colormap
) -> None:
    """
    可直接覆盖原版。改动点8已完成：
      1) try/finally 保证 return_attention 无论异常都能恢复；
      2) 注意力做“非负+归一化”，Top-k与曲线更稳定；
      3) 形状兼容 (1, m, 1, H, W) 与 (m, 1, H, W)。

    产物：
      sample{i}_topk_indices.npy / sample{i}_attn.npy
      topk_attention_slices.png
      attention_curve_sample{i}.png / .csv
      test_attention_curve.png（固定名）
    """
    import os, math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import cm as _cm

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    # --- 新增：per-slice attention CSV 写入器 ---
    tables_dir = os.path.join(out_dir, "tables")
    ensure_dir(tables_dir)
    per_slice_csv = os.path.join(tables_dir, "per_slice_attention.csv")
    fieldnames = [
        "sample_idx", "patient_id", "slice_idx", "seq_type", "filename",
        "attn_pos", "attn_weight", "is_topk", "logit", "prob", "true_label"
    ]
    csv_file = open(per_slice_csv, mode="a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not os.path.exists(per_slice_csv) or os.path.getsize(per_slice_csv) == 0:
        csv_writer.writeheader()

    # === 自动切换 return_attention（异常也能恢复） ===
    has_attr = hasattr(model, "return_attention")
    prev_flag = None
    if has_attr:
        prev_flag = model.return_attention
        model.return_attention = True

    # -------------------- 可视化小工具 --------------------
    def _denorm_for_display_zscore(x: np.ndarray) -> np.ndarray:
        """灰度→z-score→百分位截断→[0,1]→gamma增强，返回 float32"""
        x = np.asarray(x, dtype=np.float32)
        mu = float(np.mean(x)); sd = float(np.std(x)) if np.std(x) > 1e-6 else 1.0
        x = (x - mu) / sd
        lo, hi = np.percentile(x, [clip_percentile[0], clip_percentile[1]])
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
        x01 = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        return np.power(x01, max(1.0 / max(gamma, 1e-6), 1.0)).astype(np.float32)

    def _overlay_constant_heat(base01: np.ndarray, w_scalar: float) -> np.ndarray:
        """把权重 w 作为常量热图叠加到归一化后的灰度 base01 上"""
        base01 = np.clip(base01, 0.0, 1.0)
        heat01 = np.full_like(base01, float(w_scalar))
        cmap_obj = _cm.get_cmap(cmap)
        heat_rgb = cmap_obj(heat01)[..., :3]
        base_rgb = np.stack([base01, base01, base01], axis=-1)
        out = (1.0 - alpha) * base_rgb + alpha * heat_rgb
        return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)

    # -------------------- 主流程 --------------------
    model.eval()
    try:
        processed = 0
        rows = []
        for i, batch in enumerate(val_loader):
            # 兼容两种返回：(bags, labels) 或 (bags, labels, meta)
            if len(batch) == 3:
                bags, labels, meta = batch
            else:
                bags, labels = batch
                meta = None

            # 兼容 5D (1, m, 1, H, W) → 4D (m, 1, H, W)
            if bags.dim() == 5 and bags.size(0) == 1:
                bags = bags.squeeze(0)
            bags = bags.to(device=next(model.parameters()).device, dtype=torch.float32, non_blocking=True)
            if bags.dim() == 3:           # (m, H, W) → (m, 1, H, W)（以防万一）
                bags = bags.unsqueeze(1)
            outputs = model(bags)
            if not (isinstance(outputs, (tuple, list)) and len(outputs) >= 2):
                raise RuntimeError("模型 forward 必须返回 (logits, attention_weights)")

            logits, attn = outputs[0], outputs[1]

            # ---- 注意力提取 + 安全归一化（非负 + 和为1）----
            attn_np = attn.squeeze(0).detach().cpu().float().numpy()
            attn_np = np.maximum(attn_np, 1e-12)
            attn_np = attn_np / attn_np.sum()
            attn_np = attn_np.astype(np.float32)

            # bag → numpy（仅用于可视化）
            bag_np = bags.detach().cpu().numpy()          # (m, 1, H, W) 或 (m, H, W)
            if bag_np.ndim == 4 and bag_np.shape[1] == 1:
                bag_np = bag_np[:, 0, :, :]               # → (m, H, W)

            m = int(attn_np.shape[0])
            k = int(min(topk, m))
            topk_idx = np.argsort(attn_np)[-k:][::-1]
        # ---------- 新增：把 per-slice 注意力写入 CSV ----------
        with torch.no_grad():
            # logits 可能在 outputs[0]
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 1:
                logit_val = outputs[0]
            else:
                logit_val = outputs

            if isinstance(logit_val, torch.Tensor):
                if logit_val.dim() > 0:
                    logit_val = logit_val.squeeze()
                prob_val = torch.sigmoid(logit_val.float())
                try:
                    logit_scalar = float(logit_val.mean().item())
                    prob_scalar  = float(prob_val.mean().item())
                except Exception:
                    logit_scalar = float(logit_val)
                    prob_scalar  = float(prob_val)
            else:
                logit_scalar = float(logit_val)
                prob_scalar  = float(logit_val)

            if isinstance(labels, torch.Tensor):
                ytrue = labels.detach().cpu().float()
                try:
                    true_scalar = float(ytrue.mean().item())
                except Exception:
                    true_scalar = float(ytrue)
            else:
                true_scalar = float(labels)

        # meta 信息
        pid = None; seq_type = None; fname = None; slice_idx_vec = None
        if meta is not None and isinstance(meta, dict):
            pid = str(meta.get("checkno", ""))
            seq_type = meta.get("seq_type", None)
            fname = meta.get("filename", None)
            slice_idx_vec = meta.get("slice_idx", None)

        # 写入每个实例
        m = int(attn_np.shape[0])
        topk_set = set(topk_idx.tolist()) if hasattr(topk_idx, "tolist") else set(list(topk_idx))
        for inst_idx in range(m):
            is_topk = 1 if inst_idx in topk_set else 0

            if isinstance(slice_idx_vec, torch.Tensor):
                if slice_idx_vec.numel() > inst_idx:
                    cur_slice = int(slice_idx_vec.view(-1)[inst_idx].item())
                else:
                    cur_slice = inst_idx
            elif isinstance(slice_idx_vec, (list, tuple)):
                cur_slice = int(slice_idx_vec[inst_idx]) if len(slice_idx_vec) > inst_idx else inst_idx
            else:
                cur_slice = inst_idx

            csv_writer.writerow({
                "sample_idx": i,
                "patient_id": pid if pid is not None else "",
                "slice_idx": cur_slice,
                "seq_type": seq_type if seq_type is not None else "",
                "filename": fname if fname is not None else "",
                "attn_pos": inst_idx,
                "attn_weight": float(attn_np[inst_idx]),
                "is_topk": is_topk,
                "logit": logit_scalar,
                "prob": prob_scalar,
                "true_label": true_scalar
            })
        # ---------- 新增代码结束 ----------

                    # 保存原始注意力与索引
            np.save(os.path.join(out_dir, f"sample{i}_topk_indices.npy"), topk_idx)
            np.save(os.path.join(out_dir, f"sample{i}_attn.npy"), attn_np)

            # ---------- Top-k 拼图 ----------
            cols = min(len(topk_idx), 3)
            rows = int(math.ceil(len(topk_idx) / cols)) if len(topk_idx) > 0 else 1
            fig = plt.figure(figsize=(3.2 * cols, 3.2 * rows))
            for j, idx in enumerate(topk_idx):
                ax = fig.add_subplot(rows, cols, j + 1)
                img = bag_np[idx]
                if normalize_mode == "zscore":
                    base01 = _denorm_for_display_zscore(img)
                else:
                    # none: 线性归一化到 [0,1]
                    vmin, vmax = np.percentile(img, [1.0, 99.0])
                    base01 = (np.clip(img, vmin, vmax) - vmin) / max(vmax - vmin, 1e-6)
                    base01 = base01.astype(np.float32)

                out_rgb = _overlay_constant_heat(base01, attn_np[idx])
                ax.imshow(out_rgb)
                ax.set_title(f"idx={int(idx)} | w={attn_np[idx]:.6f}")
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "topk_attention_slices.png"), dpi=160)
            plt.close(fig)

            # ---------- 注意力曲线（样本）+ CSV ----------
            xs = np.arange(m, dtype=np.int32)
            fig2, ax2 = plt.subplots(figsize=(6.4, 3.2))
            ax2.plot(xs, attn_np, marker="o")
            ax2.set_xlabel("Slice idx"); ax2.set_ylabel("Attention Score")
            ax2.set_title("Attention Curve (sample)")
            ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax2.ticklabel_format(style="plain", useOffset=False, axis="y")
            ax2.set_ylim(0.0, float(attn_np.max()) * 1.10 if attn_np.size else 1.0)

            fig2.tight_layout()
            fig2.savefig(os.path.join(out_dir, f"attention_curve_sample{i}.png"), dpi=160)
            plt.close(fig2)

            pd.DataFrame({"slice_idx": xs, "attention": attn_np}).to_csv(
                os.path.join(out_dir, f"attention_curve_sample{i}.csv"),
                index=False, encoding="utf-8"
            )

            # ---------- 注意力曲线（固定文件名，用于快速查看最后一次） ----------
            fig3, ax3 = plt.subplots(figsize=(6.4, 3.2))
            ax3.plot(xs, attn_np, marker="o")
            ax3.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax3.ticklabel_format(style="plain", useOffset=False, axis="y")
            ax3.set_ylim(0.0, float(attn_np.max()) * 1.10 if attn_np.size else 1.0)

            ax3.set_xlabel("Slice idx"); ax3.set_ylabel("Attention Score")
            ax3.set_title("ABMILWithCNN - Attention Curve")
            fig3.tight_layout()
            fig3.savefig(os.path.join(out_dir, "test_attention_curve.png"), dpi=160)
            plt.close(fig3)

            processed += 1
            if processed >= 3:   # 按你原设定：仅导出前 3 个样本，可自行修改
                break

        # 若项目中有该函数，可整理输出目录结构；没有则忽略
        try:
            organize_outputs(out_dir)  # 与 for 同级（只执行一次）
        except Exception:
            pass

    finally:
        # 无论是否异常都恢复模型的 return_attention
        if has_attr and prev_flag is not None:
            model.return_attention = prev_flag

        # 关闭 CSV 文件
        try:
            csv_file.close()
        except Exception:
            pass

# ========================= 评估与出图主流程 =========================
def evaluate_and_plot(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: List[str],
    out_dir: str,
    threshold_mode: str = "youden"  # "youden" 或 "maxf1"
) -> Dict[str, Any]:
    """
    依据 y_true/y_score 计算阈值、生成预测、导出指标与科研风格图。
    会保存：
      - arrays: y_true.npy, y_score.npy, y_pred.npy, thresholds.npy
      - text:   classification_report.txt
      - tables: metrics_basic.json
      - figures: roc_curve.png, pr_curve.png, calibration_curve.png,
                 decision_curve.png, f1_scores.png, auc_per_label.png,
                 以及 figures/confusion_matrices/*.png
    """
    ensure_dir(out_dir)

    # ========= 选择阈值并二值化 =========
    if threshold_mode.lower() == "youden":
        thresholds = select_thresholds_by_youden(y_true, y_score)
    else:
        thresholds = select_thresholds_by_maxF1(y_true, y_score)
    y_pred = binarize_with_thresholds(y_score, thresholds)

    # ========= 计算指标 =========
    metrics = compute_basic_metrics(y_true, y_pred)  # 返回内含 f1_per_class 等
    try:
        auc_per_class = [roc_auc_score(y_true[:, c], y_score[:, c]) for c in range(y_true.shape[1])]
    except Exception:
        auc_per_class = [float("nan")] * y_true.shape[1]

    # ========= 导出数组与基础指标 =========
    np.save(os.path.join(out_dir, "y_true.npy"), y_true)
    np.save(os.path.join(out_dir, "y_score.npy"), y_score)
    np.save(os.path.join(out_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(out_dir, "thresholds.npy"), thresholds)
    save_json(os.path.join(out_dir, "metrics_basic.json"), metrics)

    # ========= 混淆矩阵（PNG） =========
    # 输出目录：<out_dir>/figures/confusion_matrices/confusion_matrix_<Class>.png
    save_confusion_matrices_png(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        out_dir=out_dir
    )

    # ========= 分类报告（文本） =========
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # ========= 绘图（单轴对比，科研风格） =========
    plot_roc_single_axis(
        y_true, y_score, class_names,
        out_path=os.path.join(out_dir, "roc_curve.png"),
        title="ROC Curve"
    )
    plot_pr_single_axis(
        y_true, y_score, class_names,
        out_path=os.path.join(out_dir, "pr_curve.png"),
        title="Precision-Recall Curve"
    )
    plot_calibration_single_axis(
        y_true, y_score, class_names,
        out_path=os.path.join(out_dir, "calibration_curve.png"),
        n_bins=10,
        title="Calibration Curve"
    )
    plot_dca_single_axis(
        y_true, y_score, class_names,
        out_path=os.path.join(out_dir, "decision_curve.png"),
        title="Decision Curve Analysis"
    )
    plot_bars_single_axis(
        metrics["f1_per_class"], class_names,
        out_path=os.path.join(out_dir, "f1_scores.png"),
        title="Per-label F1-score", ylabel="F1-score"
    )
    plot_bars_single_axis(
        auc_per_class, class_names,
        out_path=os.path.join(out_dir, "auc_per_label.png"),
        title="Per-label AUC", ylabel="AUC"
    )

    # ========= 结果分类整理 =========
    organize_outputs(out_dir)  # 与 return 同级，函数末尾执行一次

    return {
        "thresholds": thresholds.tolist(),
        "metrics": metrics,
        "auc_per_class": auc_per_class
    }
def train(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int,
    model_name: str,
    threshold_strategy: str = "youden",   # 或 "maxF1"
    save_raw: bool = True,
    calibrate: bool = False,              # 形参保留，当前不启用
    inject_fake_fn: bool = False,         # 形参保留，当前不启用
    result_root: str = "results",
    track_curves: bool = True,            # 已在 fit() 内生成 train_val_loss_auc.png
    use_earlystop: bool = True,           # 通过 patience 控制开关
    earlystop_monitor: str = "val_macro_auc",
    earlystop_patience: int = 10,
    earlystop_delta: float = 5e-4,
    use_amp: bool = True, 
    # 学习率/权重衰减
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
     # === 可选：注意力熵/强监督先验 ===
    lambda_entropy: float = 0.0,
    lambda_prior: float = 0.0,
    strong_lookup: Optional[dict] = None,
    pos_weight: Optional[torch.Tensor] = None,
    focal_gamma: Optional[float] = None
) -> Dict[str, Any]:

    """
    训练入口（与本文件其余函数完全适配）：
      1) 用 fit() 训练（含早停、曲线）；
      2) validate() 拿到 y_true/y_score；
      3) 阈值选择、写出 arrays/json；
      4) evaluate_and_plot() 统一绘图与混淆矩阵；
      5) 返回与旧脚本兼容的汇总字典。
    """
    # -------- 目录与优化器/损失 --------
    out_dir = os.path.join(result_root, model_name)
    ensure_dir(out_dir)
# 确保 pos_weight 在正确设备 & float 类型
    if pos_weight is not None:
       pos_weight = pos_weight.to(next(model.parameters()).device).float()
# focal_gamma <= 0 时视为禁用，强制设为 None
    if (focal_gamma is None) or (focal_gamma <= 0):
        focal_gamma = None
        
    criterion = build_loss(pos_weight=pos_weight, focal_gamma=focal_gamma)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
)


    # -------- 训练（与本文件 fit 签名一致）--------
       # -------- 训练（与本文件 fit 签名一致） --------
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,            # 直接用形参
        out_dir=out_dir,
        monitor=earlystop_monitor,
        earlystop_patience=(earlystop_patience if use_earlystop else 10**9),
        earlystop_delta=earlystop_delta,
        use_amp=use_amp,                  # 直接用形参
        device=device,
        model_name=model_name,
        fold_idx=None,
        lambda_entropy=lambda_entropy,
        lambda_prior=lambda_prior,        # 弱监督时=0，不触发先验
        strong_lookup=strong_lookup,
    )

    # === 从 fit() 的历史曲线中拿到末轮指标（供后续使用，避免 macro_auc 未定义） ===
    _val_auc_curve   = history.get("val_macro_auc", [])
    _val_loss_curve  = history.get("val_losses", [])
    _trn_loss_curve  = history.get("train_losses", [])

    macro_auc        = float(_val_auc_curve[-1])   if len(_val_auc_curve)  > 0 else float("nan")
    val_loss_last    = float(_val_loss_curve[-1])  if len(_val_loss_curve) > 0 else float("nan")
    train_loss_last  = float(_trn_loss_curve[-1])  if len(_trn_loss_curve) > 0 else float("nan")

    # -------- 完整验证，得到 y_true / y_score --------
    # （如需最严谨，可在此处先加载 best_model.pth 再 validate）
    val_loss, y_true, y_score = validate(model, val_loader, criterion)

    # === 确保 AUC 变量已定义：micro_auc +（必要时）macro_auc 兜底 ===
    yt = np.asarray(y_true)
    ys = np.asarray(y_score)

    # micro AUC：将多标签展平计算
    try:
        micro_auc = float(roc_auc_score(yt.reshape(-1), ys.reshape(-1)))
    except Exception:
        micro_auc = float("nan")

    # macro AUC 兜底：逐类 AUC 的平均（只统计正负样本都存在的类别）
    def _safe_macro_auc(yt_np: np.ndarray, ys_np: np.ndarray) -> float:
        try:
            if yt_np.ndim != 2:
                return float("nan")
            auc_list = []
            C = yt_np.shape[1]
            for ci in range(C):
                # 该列若只有 0 或只有 1，会导致 roc_auc_score 报错，跳过
                if np.unique(yt_np[:, ci]).size < 2:
                    continue
                auc_list.append(roc_auc_score(yt_np[:, ci], ys_np[:, ci]))
            return float(np.mean(auc_list)) if len(auc_list) > 0 else float("nan")
        except Exception:
            return float("nan")

    # 如果前面已从 history 拿到 macro_auc，就不覆盖；否则用兜底值
    if ("macro_auc" not in locals()) or (macro_auc is None) or (not np.isfinite(macro_auc)):
        macro_auc = _safe_macro_auc(yt, ys)

    # 标签名：优先从数据集取
    if hasattr(val_loader, "dataset") and hasattr(val_loader.dataset, "label_columns"):
        class_names = list(val_loader.dataset.label_columns)
    else:
        C = y_true.shape[1]
        class_names = [f"label_{i}" for i in range(C)]

    # -------- 阈值选择与保存 --------
    if threshold_strategy.lower() == "youden":
        thresholds = select_thresholds_by_youden(y_true, y_score)
    else:
        thresholds = select_thresholds_by_maxF1(y_true, y_score)
    np.save(os.path.join(out_dir, "thresholds.npy"), thresholds)

    # 二值化
    y_pred = binarize_with_thresholds(y_score, thresholds)

    # -------- 统一结果 JSON（final_metrics.json） --------
    metrics = compute_metrics(y_true, y_score, y_pred, label_cols=class_names)
    ap_micro  = float(metrics.get("AP_Micro", float("nan")))
    f1_macro  = float(metrics.get("F1_Macro", float("nan")))

    # 额外记录曲线末轮值，方便 sanity check
    metrics["Macro_AUC_curve_last"]   = macro_auc
    metrics["Micro_AUC_curve_last"]   = micro_auc
    metrics["Val_Loss_curve_last"]    = val_loss_last
    metrics["Train_Loss_curve_last"]  = train_loss_last

    save_json(os.path.join(out_dir, "final_metrics.json"), metrics)

    # -------- 保存 preds/labels（供 DeLong 检验） --------
    np.savez_compressed(
        os.path.join(out_dir, "fold_preds_labels.npz"),
        labels=y_true,
        preds=y_score,
    )


    # -------- 评估与科研风格出图（含混淆矩阵）--------
    evaluate_and_plot(
        y_true=y_true,
        y_score=y_score,
        class_names=class_names,
        out_dir=out_dir,
        threshold_mode=("youden" if threshold_strategy.lower() == "youden" else "maxf1"),
    )
    # -------- 额外导出 CSV（总览 + 分类别）--------
    # 1) 总览：summary_metrics.csv（单行）
    summary_row = {
        "model": model_name,
        "Macro_AUC": macro_auc,
        "Micro_AUC": micro_auc,
        "AP_Micro": ap_micro,
        "F1_Macro": f1_macro,
    }
    pd.DataFrame([summary_row]).to_csv(
        os.path.join(out_dir, "summary_metrics.csv"),
        index=False, encoding="utf-8"
    )

    # 2) 分类别：per_class_metrics.csv（逐类 1 行）
    per_class_rows = []
    for i, name in enumerate(class_names):
    # —— 这里替换原来的 AUC 取值 —— #
        try:
            if np.unique(y_true[:, i]).size > 1:   # 该类在 val 有正有负才可算 AUC
                auc_i = float(roc_auc_score(y_true[:, i], y_score[:, i]))
            else:
                auc_i = float("nan")
        except Exception:
            auc_i = float("nan")

        per_class_rows.append({
            "class":     name,
            "AUC":       auc_i,   # ← 用刚算好的 auc_i
            "F1":        float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "Precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "Recall":    float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "Threshold": float(thresholds[i]),
        })

    pd.DataFrame(per_class_rows).to_csv(
        os.path.join(out_dir, "per_class_metrics.csv"),
        index=False, encoding="utf-8"
    )

    # 把刚生成的 CSV 也归档到 tables/ 目录下（与其它产物风格一致）
    try:
        organize_outputs(out_dir)
    except Exception:
        pass

    # -------- 与旧脚本兼容的返回字段 --------
    # AUC
    try:
        macro_auc = float(roc_auc_score(y_true, y_score, average="macro"))
    except Exception:
        macro_auc = float("nan")
    try:
        micro_auc = float(roc_auc_score(y_true, y_score, average="micro"))
    except Exception:
        micro_auc = float("nan")
    try:
        ap_micro = float(average_precision_score(y_true, y_score, average="micro"))
    except Exception:
        ap_micro = float("nan")

        # F1 (使用上面二值化结果)
    try:
        f1_macro = float(np.mean([f1_score(y_true[:, c], y_pred[:, c], zero_division=0) 
                                  for c in range(y_true.shape[1])]))
    except Exception:
        f1_macro = float("nan")

    # 每类 AUC (保持旧键名风格)
    auc_per_label = {}
    for i, name in enumerate(class_names):
        try:
            auc_i = float(roc_auc_score(y_true[:, i], y_score[:, i]))
        except Exception:
            auc_i = float("nan")
        auc_per_label[f"AUC_{name}"] = auc_i

    # --- 确保变量一定被定义（避免 UnboundLocalError） ---
    macro_auc = locals().get("macro_auc", float("nan"))
    micro_auc = locals().get("micro_auc", float("nan"))
    ap_micro  = locals().get("ap_micro", float("nan"))
    f1_macro  = locals().get("f1_macro", float("nan"))

    # 组织返回结果
    ret = {
        "Macro_AUC": macro_auc,
        "Micro_AUC": micro_auc,
        "AP_Micro": ap_micro,
        "F1_Macro": f1_macro,
    }
    ret.update(auc_per_label)
    return ret

            # 混淆曲线
def save_confusion_matrices_png(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                class_names,
                                out_dir: str) -> None:
    """
    将每个类别的一对多（二分类）混淆矩阵保存为 PNG 热力图。
    输出目录：<out_dir>/figures/confusion_matrices/confusion_matrix_<Class>.png
    说明：
      - y_true, y_pred 形状均为 [N, C]，二值 {0,1}
      - class_names 为长度 C 的类别名称列表
      - 本函数只导出 PNG；如需 CSV 可另外添加保存逻辑
    """
    # 目标目录：figures/confusion_matrices
    fig_root = os.path.join(out_dir, "figures")
    dest_dir = os.path.join(fig_root, "confusion_matrices")
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(fig_root, exist_ok=True)

    # 科研风格
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    C = y_true.shape[1]
    for c in range(C):
        yt = y_true[:, c].astype(int)
        yp = y_pred[:, c].astype(int)

        # [[TN, FP], [FN, TP]]
        cm = confusion_matrix(yt, yp, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"Confusion Matrix - {class_names[c]}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])

        # 数值标注
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]),
                        ha="center", va="center",
                        color="black", fontsize=10)

        fig.tight_layout()
        out_png = os.path.join(dest_dir, f"confusion_matrix_{class_names[c]}.png")
        fig.savefig(out_png)
        plt.close(fig)

# ========================= 使用示例（供参考，可删） =========================
if __name__ == "__main__":
    # 示意：不要在正式训练脚本中运行此块
    set_seed(42)
    ensure_dir("./_demo_outputs")

    # 构造假数据演示
    N, C = 200, 3
    rng = np.random.default_rng(0)
    y_true_demo = rng.integers(0, 2, size=(N, C))
    y_score_demo = np.clip(y_true_demo * 0.6 + rng.normal(0.3, 0.2, size=(N, C)), 0, 1)

    evaluate_and_plot(y_true_demo, y_score_demo, ["Label_0", "Label_1", "Label_2"], out_dir="./_demo_outputs")

    # 演示训练-验证曲线
    train_loss_demo = list(np.linspace(0.6, 0.3, 20) + rng.normal(0, 0.01, 20))
    val_loss_demo = list(np.linspace(0.62, 0.33, 20) + rng.normal(0, 0.015, 20))
    val_auc_demo = list(np.linspace(0.55, 0.68, 20) + rng.normal(0, 0.01, 20))
    plot_train_val_loss_auc_single_axis(train_loss_demo, val_loss_demo, val_auc_demo,
                                        out_path="./_demo_outputs/train_val_loss_auc.png",
                                        title="Train/Val Loss & Val Macro AUC")
