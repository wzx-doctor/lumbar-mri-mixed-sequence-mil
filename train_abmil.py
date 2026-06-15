# -*- coding: utf-8 -*-
# train_abmil.py
# 全流程：k-fold + bag_size 消融 + 统计学（Bootstrap CI / 配对 t 检验 / DeLong）
# 新增：--models 选择器（未设置时默认跑 ABMIL,MaxPoolingMIL,MeanPoolingMIL,GatedTempAttentionMIL）
# 依赖：abmil_dataset.py, abmil_model.py, train_utils.py

import os
import argparse
import json
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import ttest_rel
from scipy.stats import norm
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from abmil_dataset import ABMILDataset
# 原有导入：ABMILWithCNN, MaxPoolingMIL, MeanPoolingMIL
from abmil_model import ABMILWithCNN, MaxPoolingMIL, MeanPoolingMIL
# 新增：门控温度注意力 MIL（需要在 abmil_model.py 中提供该类）
from abmil_model import GatedTempAttentionMIL  # ← 与后续包保持一致
from train_utils import train, test_attention_output
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)

# -----------------------
# Bootstrap 95% CI
# -----------------------
def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = SEED) -> Tuple[float, float]:
    """对折级指标做非参数自助法 CI。"""
    values = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(values, size=len(values), replace=True)
        boots.append(np.mean(samp))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi
# -----------------------
# DeLong test (AUC difference, per-class)
# -----------------------
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def _fast_delong(y_true: np.ndarray, y_scores: np.ndarray):
    y_true = np.array(y_true, dtype=int)
    y_scores = np.array(y_scores, dtype=float)
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return np.nan, np.nan
    tx = _compute_midrank(np.concatenate([pos, neg]))
    tpos = tx[:m]
    tneg = tx[m:]
    auc = (tpos.sum() - m * (m + 1) / 2.0) / (m * n)
    v01 = (tpos - (m + 1) / 2.0) / n
    v10 = 1.0 - (tneg - (n + 1) / 2.0) / m
    s01 = np.var(v01, ddof=1)
    s10 = np.var(v10, ddof=1)
    var = (s01 / m) + (s10 / n)
    return auc, var

def delong_p_value(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    auc_a, var_a = _fast_delong(y_true, pred_a)
    auc_b, var_b = _fast_delong(y_true, pred_b)
    if np.isnan(auc_a) or np.isnan(auc_b):
        return np.nan
    z = (auc_a - auc_b) / np.sqrt(var_a + var_b + 1e-12)
    p = 2 * (1 - norm.cdf(np.abs(z)))
    return float(p)
# -----------------------
# 注意力诊断：熵（normalized entropy）
# -----------------------
@torch.inference_mode()
def compute_attention_entropy(
    model: nn.Module,
    val_loader: DataLoader,
    out_dir: str,
    topk: int = 6,
) -> None:
    """
    在验证集逐样本计算注意力权重的归一化熵：H(p)/log(m)，m 为 bag 内切片数。
    - 导出 attention_entropy.csv（含样本索引、bag 尺寸、熵、top-k 均值等）
    - 不依赖 train_utils 的可视化流程，可与 test_attention_output 并行存在
    """
    os.makedirs(out_dir, exist_ok=True)
    # 自动切换 return_attention
    has_attr = hasattr(model, "return_attention")
    prev_flag = None
    if has_attr:
        prev_flag = model.return_attention
        model.return_attention = True

    rows = []
    for i, batch in enumerate(val_loader):
        # ---------- 统一解包：兼容 (bags, labels, meta) / (bags, labels) / 其它嵌套 ----------
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 3:
                bags, labels, meta = batch[0], batch[1], batch[2]
            elif len(batch) == 2:
                bags, labels = batch
                meta = None
            else:
                bags, labels, meta = batch[0], None, None
        else:
            bags, labels, meta = batch, None, None

        # ---------- 统一设备/精度/形状 ----------
        device = next(model.parameters()).device
        bags = bags.to(device=device, dtype=torch.float32, non_blocking=True)
        # 有时 DataLoader 给 (1, m, 1, H, W)；也可能是 (m, H, W)
        if bags.ndim == 5 and bags.size(0) == 1:
            bags = bags.squeeze(0)
        if bags.ndim == 3:
            bags = bags.unsqueeze(1)  # (m, H, W) -> (m, 1, H, W)

        # ---------- 前向 ----------
        out = model(bags)  # 期望返回：(logits, w[, H]) 当 return_attention=True

        if isinstance(out, (list, tuple)) and len(out) >= 2:
            _, w = out[0], out[1]
        else:
            # 不支持注意力返回时跳过
            continue

        w = w.squeeze(0).detach().cpu().float().numpy()
        w = np.clip(w, 1e-12, 1.0)
        p = w / w.sum()
        m = float(len(p))
        ent = float(-(p * np.log(p)).sum() / (np.log(m) + 1e-12))

        topk_idx = np.argsort(p)[-min(topk, len(p)):]
        topk_mean = float(p[topk_idx].mean())
        rows.append({
            "sample_index": i,
            "bag_size": int(len(p)),
            "entropy_norm": ent,
            "topk": int(min(topk, len(p))),
            "topk_mean": topk_mean,
        })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "attention_entropy.csv"), index=False, encoding="utf-8")

    # 还原 return_attention
    if has_attr:
        model.return_attention = prev_flag
# -----------------------
# argparse
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="ABMIL/Max/Mean/GatedTempAttentionMIL：k-fold + Bootstrap CI + DeLong + 消融（bag size）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 数据与标签
    p.add_argument("--csv", type=str, default="D:/MyProject/LDH_matched_labels.csv",help="包含 checkno 与标签列的 CSV（病例级标签）")
    p.add_argument("--npy_csv", type=str, default="D:/RawMRI/dataset_npy_v4_full/npy_labels.csv",help="NPY 清单 CSV（至少含列：checkno,npy_path；可含 seq_type）")
    p.add_argument("--data_dir", type=str, default="D:/RawMRI/dataset_npy_v4_full",help="npy 图像目录（与 npy_path 所在位置一致即可）")
    p.add_argument("--label_cols", type=str, default="L3/4,L4/5,L5/S1")
    # 只用某一序列或两者都用
    p.add_argument(
        "--only_seq",
        type=str,
        default="both",
        choices=["both", "T2", "T2-FS"],
        help="训练/评估时仅使用 T2 或 T2-FS；默认 both 表示两者都用"
    )
    # 日志：Dataset/构建阶段是否详细打印
    p.add_argument("--verbose", dest="verbose", action="store_true", help="开启详细日志")
    p.add_argument("--quiet",   dest="verbose", action="store_false", help="关闭详细日志")
    p.set_defaults(verbose=False)
    # 是否导出注意力可视化与熵统计（val 上）
    p.add_argument("--export_attention",     dest="export_attention", action="store_true",
                  help="导出注意力图与 attention_entropy.csv（默认开启）")
    p.add_argument("--no_export_attention",  dest="export_attention", action="store_false",
                  help="不导出注意力相关文件")
    p.set_defaults(export_attention=True)

    # 训练设置
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1, help="训练阶段需保持 1（实现依赖 squeeze(0)）")
    p.add_argument("--hidden_dim", type=int, default=128)

    # 消融：bag size 与取片策略
    p.add_argument("--bag_sizes", type=str, default="10")
    p.add_argument("--slice_sampling", type=str, default="center", choices=["center", "uniform", "random"])

    # 早停/曲线/阈值/校准
    p.add_argument("--use_earlystop", dest="use_earlystop", action="store_true", help="启用 EarlyStopping（默认已启用）")
    p.add_argument("--no_use_earlystop", dest="use_earlystop", action="store_false", help="关闭 EarlyStopping")
    p.set_defaults(use_earlystop=True)

    p.add_argument("--earlystop_monitor", type=str, default="val_macro_auc", choices=["val_macro_auc", "val_loss"])
    p.add_argument("--earlystop_patience", type=int, default=10)
    p.add_argument("--earlystop_delta", type=float, default=5e-4)

    p.add_argument("--track_curves", dest="track_curves", action="store_true", help="保存 train/val 曲线（默认已启用）")
    p.add_argument("--no_track_curves", dest="track_curves", action="store_false", help="不保存训练曲线")
    p.set_defaults(track_curves=True)

    p.add_argument("--threshold_strategy", type=str, default="youden", choices=["youden", "maxF1"])
    p.add_argument("--calibrate", action="store_true", help="是否启用概率校准")
    p.add_argument("--calibrator_type", type=str, default="isotonic", choices=["isotonic", "platt"])
    p.add_argument("--inject_fake_fn", action="store_true", help="Debug 伪阴性（默认关闭）")
    p.add_argument("--results_root", type=str, default="results", help="所有折结果写到该目录下")

    # 模型选择器
    p.add_argument("--models",type=str,default="GatedTempAttentionMIL",help=(
            "选择要运行的模型列表，英文逗号分隔；大小写需与类名一致。"
            "示例：--models ABMIL 或 --models \"ABMIL,MeanPoolingMIL,GatedTempAttentionMIL\"。"
            "未设置时默认跑 ABMIL,MaxPoolingMIL,MeanPoolingMIL,GatedTempAttentionMIL。这个是四个都跑：ABMIL,MaxPoolingMIL,MeanPoolingMIL,GatedTempAttentionMIL"
        ),
    )

    # —— 新增：门控温度注意力超参（写入 run_config.json，以便追溯）
    p.add_argument("--attn_dim", type=int, default=128, help="注意力网络的隐层/投影维度")
    p.add_argument("--init_tau", type=float, default=0.7, help="softmax 温度初值（>0），推理与训练将使用可学习温度")
    p.add_argument("--lambda_entropy", type=float, default=0.0, help="注意力熵正则（集成到模型/训练时生效）")
    p.add_argument("--attn_dropout", type=float, default=0.30,help="Dropout prob on attention logits before softmax")
    # -- 新增：专管 A 的 dropout（softmax 前 logits 的不是这个）
    p.add_argument("--a_dropout",type=float,default=None,   # 建议默认 None，表示未显式设置时走 --attn_dropout，help="Dropout prob on attention A (feature side). If None, fallback to --attn_dropout."
)
    # ==== 强监督：强标签路径与先验权重（步骤 2.2）====
    p.add_argument("--strong_dir", type=str, default=None,
                help="强监督图像/项目目录（可选，仅记录路径，代码不直接使用）")
    # 在 parse_args() 内（你当前文件已有这两行，只改 default）         default=r"D:/RawMRI/StrongLS/strong_labels.csv"
    p.add_argument("--strong_csv", type=str,
        default=r"D:/RawMRI/StrongLS/strong_labels.csv",
        help="强监督 slice 级标注 CSV")

    p.add_argument("--lambda_prior", type=float,
        default=0.2,   # 默认开启强监督先验default=0.2,
        help="注意力先验损失系数")

    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.results_root, exist_ok=True)
    if args.a_dropout is None:
       args.a_dropout = args.attn_dropout
    # ===== 构建 strong_lookup：(checkno, slice_idx) -> multi-hot 标签 =====
    # ===== 构建 strong_lookup：支持 (checkno, seq_type, slice_idx) 或 (checkno, slice_idx) =====
    strong_lookup = None
    if args.lambda_prior > 0 and args.strong_csv and os.path.isfile(args.strong_csv):
        s = pd.read_csv(args.strong_csv)
        wanted_cols = ["L3/4_has_box", "L4/5_has_box", "L5/S1_has_box"]
        cols = [c for c in wanted_cols if c in s.columns]
        if len(cols) == 0:
            raise ValueError(f"strong_csv 中未发现任何标签列，期望列之一：{wanted_cols}")
        if "checkno" not in s.columns or "slice_idx" not in s.columns:
            raise ValueError("strong_csv 必须包含列 'checkno' 与 'slice_idx'")

        # 统一格式
        s["checkno"] = s["checkno"].astype(str).str.strip()
        s["slice_idx"] = s["slice_idx"].astype(int)
        if "seq_type" in s.columns:
            s["seq_type"] = (
                s["seq_type"].astype(str).str.strip().str.upper()
                .str.replace("-", "", regex=False)
                .str.replace("_", "", regex=False)
            )
            # 现在写成 T2 或 T2FS（无破折号/下划线）

        strong_lookup = {}
        for _, r in s.iterrows():
            pid = str(r["checkno"])
            slc = int(r["slice_idx"])
            vec = [int(r.get(c, 0)) for c in cols]
            if "seq_type" in s.columns and str(r["seq_type"]).strip():
                key = (pid, str(r["seq_type"]), slc)  # (checkno, 'T2'/'T2FS', idx)
            else:
                key = (pid, slc)                      # 回退键
            strong_lookup[key] = vec

        # 仅当 strong_lookup 存在时打印样例
        print(f"[INFO] strong_lookup built, total={len(strong_lookup)} keys")
        for k in list(strong_lookup.keys())[:10]:
            print("[LOOKUP sample]", k, "->", strong_lookup[k])

    elif args.lambda_prior > 0:
        print(f"[WARN] --lambda_prior={args.lambda_prior} 但 strong_csv 不存在：{args.strong_csv}，已关闭强监督先验。")
        args.lambda_prior = 0.0


    # 保存当前配置快照（便于复现）
    cfg_path = os.path.join(args.results_root, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"[INFO] 写入 {cfg_path}", flush=True)

    # 选择器解析（未设置时默认多模型全跑）
    models_to_run: List[str] = [m.strip() for m in args.models.split(",") if m.strip()]
    # 原有：{"ABMIL","MaxPoolingMIL","MeanPoolingMIL"}；现扩展加入 GatedTempAttentionMIL
    valid_models = {"ABMIL", "MaxPoolingMIL", "MeanPoolingMIL", "GatedTempAttentionMIL"}
    for m in models_to_run:
        if m not in valid_models:
            raise ValueError(f"--models 包含未知项：{m}，可选 {sorted(valid_models)}")
    print(f"[INFO] strong_lookup built, total={len(strong_lookup)} keys")
    for k in list(strong_lookup.keys())[:10]:
        print("[LOOKUP sample]", k, "->", strong_lookup[k])

    # 打印 strong_lookup 的 key 样子
    print(f"[INFO] strong_lookup built, total={len(strong_lookup)} keys")
    for i, k in enumerate(list(strong_lookup.keys())[:10]):
        print("[LOOKUP sample]", k, "->", strong_lookup[k])
# === 诊断：强标签覆盖画像 ===
    uniq_chk = {k[0] for k in strong_lookup.keys()}
    print(f"[PRIOR] unique checkno in strong_csv: {len(uniq_chk)}")
    # seq_type 分布
    seq_cnt = {}
    for k in strong_lookup.keys():
        if len(k) == 3:
            seq_cnt[k[1]] = seq_cnt.get(k[1], 0) + 1
    print("[PRIOR] seq_type counts:", dict(sorted(seq_cnt.items(), key=lambda x: -x[1])))

    # 读入标签 CSV
   # ===== 读入并合并：病例级标签（args.csv） + NPY 清单（args.npy_csv） =====

    # 1) 病例级标签（weak labels）
    if args.csv.lower().endswith(".xlsx"):
        df_label = pd.read_excel(args.csv)
    else:
        df_label = pd.read_csv(args.csv, encoding="utf-8")
    label_columns = [c.strip() for c in args.label_cols.split(",")]
    assert "checkno" in df_label.columns, "CSV 中缺少 checkno 列"
    assert all(c in df_label.columns for c in label_columns), "标签列不存在于 CSV 中"
    # 2) NPY 清单（npy_labels.csv），至少包含：checkno, npy_path；可选：seq_type
    df_npy = pd.read_csv(args.npy_csv)
    assert "checkno" in df_npy.columns and "npy_path" in df_npy.columns, \
        "npy_csv 必须至少包含列：checkno, npy_path"
    # 3) 合并：每条样本 = (checkno, [seq_type], npy_path) + 病例级标签
    df = df_npy.merge(df_label[["checkno"] + label_columns], on="checkno", how="inner").copy()
    # （可选）如果只想训练某一序列，可开启这一行：
        # === 根据 --only_seq 过滤（需 df 中存在 seq_type 列；清洗脚本已写入） ===
    if "seq_type" in df.columns and args.only_seq != "both":
        target_seq = "T2-FS" if args.only_seq.upper() == "T2-FS" else "T2"
        before = len(df)
        df = df[df["seq_type"].astype(str).str.upper() == target_seq].copy()
        print(f"[INFO] only_seq={target_seq} 过滤后样本 {len(df)}/{before}", flush=True)

    # KFold
    sgkf = StratifiedGroupKFold(n_splits=args.kfold, shuffle=True, random_state=SEED)
    # ===== 构造分层标签 y_strat 与 分组 groups =====
    # y_strat: 单列标签 -> 直接取；多列多标签 -> 做位编码(bit-pack)成一个整数，保证分层生效
    y_multi = df[label_columns].astype(int).values  # 形状 (N, C)
    if y_multi.ndim == 1 or y_multi.shape[1] == 1:
       y_strat = y_multi.reshape(-1)
    else:
    # 多标签：把每行的 0/1 向量编码成一个整数用于分层
       y_strat = (y_multi * (1 << np.arange(y_multi.shape[1]))).sum(axis=1).astype(int)

    # groups: 以病人/检查号为分组键，确保同一人不跨折
    groups = df["checkno"].astype(str).values

    BAG_SIZES = [int(x) for x in args.bag_sizes.split(",")]

    metrics_all: Dict[str, List[Dict[str, Any]]] = {}

    for m in BAG_SIZES:
        # 初始化容器
        for key in ["ABMIL", "MaxPoolingMIL", "MeanPoolingMIL", "GatedTempAttentionMIL"]:
            if key in models_to_run:
                metrics_all[f"{key}_bag{m}"] = []

        # k 折
        for fold, (tr_idx, va_idx) in tqdm(
                enumerate(sgkf.split(df, y=y_strat, groups=groups), 1),
                total=args.kfold,
                desc=f"[bag{m}] Folds"):

            print(f"\n========== bag{m} | Fold {fold}/{args.kfold} ==========")
            train_df, val_df = df.iloc[tr_idx], df.iloc[va_idx]

            # 数据
            train_ds = ABMILDataset(
                args.data_dir, train_df, label_columns,
                bag_size=m, slice_sampling=args.slice_sampling,
                return_meta=True,            # ← 训练集开启 meta
                verbose=args.verbose,        # ← 新增：把命令行开关传进去
            )
            val_ds = ABMILDataset(
                args.data_dir, val_df, label_columns,
                bag_size=m, slice_sampling=args.slice_sampling,
                return_meta=True,            # ← 验证集也开启 meta
                verbose=args.verbose,        # ← 新增
            )

            train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)

            # -----------------------
            # ABMIL
            # -----------------------
            if "ABMIL" in models_to_run:
                abmil = ABMILWithCNN(
                hidden_dim=args.hidden_dim, 
                output_dim=len(label_columns),
                a_dropout=args.a_dropout,
                attn_dropout=args.attn_dropout, 
                return_attention=True
                )
                abmil_name = f"ABMIL_bag{m}_fold{fold}"
                abmil_metrics = train(
                    abmil,
                    train_loader,
                    val_loader,
                    num_epochs=args.epochs,
                    model_name=abmil_name,
                    threshold_strategy=args.threshold_strategy,
                    save_raw=True,
                    calibrate=args.calibrate,
                    inject_fake_fn=args.inject_fake_fn,
                    result_root=args.results_root,
                    track_curves=args.track_curves,
                    use_earlystop=args.use_earlystop,
                    earlystop_monitor=args.earlystop_monitor,
                    earlystop_patience=args.earlystop_patience,
                    earlystop_delta=args.earlystop_delta,
                    lambda_prior=args.lambda_prior,
                    strong_lookup=strong_lookup,
                )
                # 注意力可视化 + 熵诊断（验证集）
                if args.export_attention:
                    test_attention_output(
                        abmil, val_loader,
                        os.path.join(args.results_root, abmil_name)
                    )
                    compute_attention_entropy(
                        abmil, val_loader,
                        os.path.join(args.results_root, abmil_name)
                    )

                if abmil_metrics:
                    # 早停元信息回填
                    es_meta = os.path.join(args.results_root, abmil_name, "early_stopping_meta.json")
                    if os.path.exists(es_meta):
                        with open(es_meta, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        abmil_metrics["ES_monitor"] = meta.get("monitor", "")
                        abmil_metrics["ES_best_epoch"] = meta.get("best_epoch", -1)
                        abmil_metrics["ES_best_score"] = meta.get("best_score", float("nan"))
                    abmil_metrics["Fold"] = fold
                    metrics_all[f"ABMIL_bag{m}"].append(abmil_metrics)

            # -----------------------
            # MaxPoolingMIL
            # -----------------------
            if "MaxPoolingMIL" in models_to_run:
                maxmil = MaxPoolingMIL(input_dim=args.hidden_dim, hidden_dim=args.hidden_dim, output_dim=len(label_columns))
                max_name = f"MaxPoolingMIL_bag{m}_fold{fold}"
                max_metrics = train(
                    maxmil,
                    train_loader,
                    val_loader,
                    num_epochs=args.epochs,
                    model_name=max_name,
                    threshold_strategy=args.threshold_strategy,
                    save_raw=True,
                    calibrate=args.calibrate,
                    inject_fake_fn=False,
                    result_root=args.results_root,
                    track_curves=args.track_curves,
                    use_earlystop=args.use_earlystop,
                    earlystop_monitor=args.earlystop_monitor,
                    earlystop_patience=args.earlystop_patience,
                    earlystop_delta=args.earlystop_delta,
                )
                if max_metrics:
                    es_meta = os.path.join(args.results_root, max_name, "early_stopping_meta.json")
                    if os.path.exists(es_meta):
                        with open(es_meta, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        max_metrics["ES_monitor"] = meta.get("monitor", "")
                        max_metrics["ES_best_epoch"] = meta.get("best_epoch", -1)
                        max_metrics["ES_best_score"] = meta.get("best_score", float("nan"))
                    max_metrics["Fold"] = fold
                    metrics_all[f"MaxPoolingMIL_bag{m}"].append(max_metrics)
            # -----------------------
            # MeanPoolingMIL
            # -----------------------
            if "MeanPoolingMIL" in models_to_run:
                meanmil = MeanPoolingMIL(input_dim=args.hidden_dim, hidden_dim=args.hidden_dim, output_dim=len(label_columns))
                mean_name = f"MeanPoolingMIL_bag{m}_fold{fold}"
                mean_metrics = train(
                    meanmil,
                    train_loader,
                    val_loader,
                    num_epochs=args.epochs,
                    model_name=mean_name,
                    threshold_strategy=args.threshold_strategy,
                    save_raw=True,
                    calibrate=args.calibrate,
                    inject_fake_fn=False,
                    result_root=args.results_root,
                    track_curves=args.track_curves,
                    use_earlystop=args.use_earlystop,
                    earlystop_monitor=args.earlystop_monitor,
                    earlystop_patience=args.earlystop_patience,
                    earlystop_delta=args.earlystop_delta,
                )
                if mean_metrics:
                    es_meta = os.path.join(args.results_root, mean_name, "early_stopping_meta.json")
                    if os.path.exists(es_meta):
                        with open(es_meta, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        mean_metrics["ES_monitor"] = meta.get("monitor", "")
                        mean_metrics["ES_best_epoch"] = meta.get("best_epoch", -1)
                        mean_metrics["ES_best_score"] = meta.get("best_score", float("nan"))
                    mean_metrics["Fold"] = fold
                    metrics_all[f"MeanPoolingMIL_bag{m}"].append(mean_metrics)
            # -----------------------
            # GatedTempAttentionMIL（新增）
            # -----------------------
                 # -----------------------
            # GatedTempAttentionMIL（新增）
            # -----------------------
            if "GatedTempAttentionMIL" in models_to_run:
                gated = GatedTempAttentionMIL(
                    input_dim=args.hidden_dim,
                    attn_dim=args.attn_dim,
                    num_classes=len(label_columns),
                    init_tau=args.init_tau,
                    dropout=args.a_dropout,
                    attn_dropout=args.attn_dropout,
                    return_attention=True
                )
                gated_name = f"GatedTempAttentionMIL_bag{m}_fold{fold}"
                gated_metrics = train(
                    gated,
                    train_loader,
                    val_loader,
                    num_epochs=args.epochs,
                    model_name=gated_name,
                    threshold_strategy=args.threshold_strategy,
                    save_raw=True,
                    calibrate=args.calibrate,
                    inject_fake_fn=False,
                    result_root=args.results_root,
                    track_curves=args.track_curves,
                    use_earlystop=args.use_earlystop,
                    earlystop_monitor=args.earlystop_monitor,
                    earlystop_patience=args.earlystop_patience,
                    earlystop_delta=args.earlystop_delta,
                    lambda_prior=args.lambda_prior,   # 弱监督时=0，不会触发先验
                    strong_lookup=strong_lookup,
                )
                # 注意力可视化 + 熵诊断（验证集）
                if args.export_attention:
                    test_attention_output(
                        gated, val_loader,
                        os.path.join(args.results_root, gated_name)
                    )
                    compute_attention_entropy(
                        gated, val_loader,
                        os.path.join(args.results_root, gated_name)
                    )

                if gated_metrics:
                    # 早停元信息回填
                    es_meta = os.path.join(args.results_root, gated_name, "early_stopping_meta.json")
                    if os.path.exists(es_meta):
                        with open(es_meta, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        gated_metrics["ES_monitor"] = meta.get("monitor", "")
                        gated_metrics["ES_best_epoch"] = meta.get("best_epoch", -1)
                        gated_metrics["ES_best_score"] = meta.get("best_score", float("nan"))
                    gated_metrics["Fold"] = fold
                    metrics_all[f"GatedTempAttentionMIL_bag{m}"].append(gated_metrics)


    # -----------------------
    # 汇总表：mean±std + 95%CI
    # -----------------------
    rows: List[Dict[str, Any]] = []
    for key, folds in metrics_all.items():
        if not folds:
            continue
        row: Dict[str, Any] = {"Model": key}
        for met in ["Macro_AUC", "Micro_AUC", "AP_Micro", "AUC_L3/4", "AUC_L4/5", "AUC_L5/S1", "F1_Macro"]:
            vals = np.array([d.get(met, np.nan) for d in folds], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            row[f"{met}_mean"] = float(np.mean(vals))
            row[f"{met}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            lo, hi = bootstrap_ci(vals)
            row[f"{met}_CI_low"] = lo
            row[f"{met}_CI_high"] = hi
        rows.append(row)
    out_csv = os.path.join(args.results_root, "metrics_summary_kfold.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[INFO] 写入 {out_csv}", flush=True)

    # -----------------------
    # 显著性检验：配对 t 检验（Macro_AUC）
    # -----------------------
    pairs = []
    for m in BAG_SIZES:
        chosen = [md for md in ["ABMIL", "MaxPoolingMIL", "MeanPoolingMIL", "GatedTempAttentionMIL"] if md in models_to_run]
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                pairs.append((f"{chosen[i]}_bag{m}", f"{chosen[j]}_bag{m}"))

    t_rows: List[Dict[str, Any]] = []
    for A, B in pairs:
        a_vals = [d.get("Macro_AUC", np.nan) for d in metrics_all.get(A, [])]
        b_vals = [d.get("Macro_AUC", np.nan) for d in metrics_all.get(B, [])]
        a_vals = np.array([x for x in a_vals if not np.isnan(x)])
        b_vals = np.array([x for x in b_vals if not np.isnan(x)])
        if len(a_vals) != len(b_vals) or len(a_vals) == 0:
            continue
        _, p_val = ttest_rel(a_vals, b_vals)
        t_rows.append({"metric": "Macro_AUC", "modelA": A, "modelB": B, "p_ttest": float(p_val)})
    # -----------------------
    # DeLong：逐类 AUC 差异
    # -----------------------
    def _load_fold_npz(model_key_prefix: str):
        lbl_all, pred_all = None, None
        for fold in range(1, args.kfold + 1):
            path = os.path.join(args.results_root, f"{model_key_prefix}_fold{fold}", "fold_preds_labels.npz")
            if not os.path.exists(path):
                continue
            data = np.load(path)
            lbl, pred = data["labels"], data["preds"]
            if lbl_all is None:
                lbl_all, pred_all = lbl, pred
            else:
                lbl_all = np.vstack([lbl_all, lbl])
                pred_all = np.vstack([pred_all, pred])
        return lbl_all, pred_all

    p_rows: List[Dict[str, Any]] = []
    label_names = [c.strip() for c in args.label_cols.split(",")]
    for A, B in pairs:
        yA, pA = _load_fold_npz(A)
        yB, pB = _load_fold_npz(B)
        if yA is None or yB is None:
            continue
        if yA.shape != yB.shape or pA.shape != pB.shape:
            print(f"[WARN] DeLong: 形状不一致，跳过 {A} vs {B}")
            continue
        for i, lab in enumerate(label_names):
            pval = delong_p_value(yA[:, i], pA[:, i], pB[:, i])  # A vs B
            p_rows.append({"metric": f"AUC_{lab}", "modelA": A, "modelB": B, "p_delong": float(pval)})
    # 合并导出 p 值
    p_df = pd.DataFrame(p_rows)
    if t_rows:
        p_df = p_df.merge(pd.DataFrame(t_rows), how="outer")
    p_path = os.path.join(args.results_root, "p_values.csv")
    p_df.to_csv(p_path, index=False, encoding="utf-8")
    print(f"[INFO] 写入 {p_path}", flush=True)
if __name__ == "__main__":
    main()
