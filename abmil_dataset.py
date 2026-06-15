# -*- coding: utf-8 -*-
"""
abmil_dataset.py  —— 纯净数据集实现（不做采样策略切换、不做归一化、不做增强）
符合以下约束：
1) 不改变像素分布：不做 z-score / min-max / gamma / 噪声 / 亮度对比度处理。
2) 统一取片：仅采用“中心窗口”索引；若切片数不足 bag_size，循环补齐。
3) 支持两类存储：
   A) 单文件：data_dir/{checkno}_img.npy -> (S,H,W) 或 (H,W)
   B) 文件夹：data_dir/{checkno}/*.npy   -> 各切片 (H,W) 或 (1,H,W) 或 (H,W,1)
4) 与训练/可视化配套：
   __getitem__ 返回 (bags, labels)，其中 bags 形状为 (m, 1, H, W)。

保留原构造签名的兼容参数（slice_sampling/normalize/augment 等），内部忽略并给出一次性警告。
"""

from __future__ import annotations
import os
import glob
import warnings
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd  # 用于加载 strong_labels.csv
from PIL import Image  # 用于加载图像
# ============================== 基础工具 ==============================

def _safe_load_npy(path: str) -> Optional[np.ndarray]:
    """稳健读取 .npy；失败返回 None。"""
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return None

def _probe_npy_shape(path: str):
    """
    只读取 .npy 的 shape，不加载全部数据。
    返回 (S,H,W) 或 (1,H,W)，失败返回 None。
    """
    try:
        a = np.load(path, mmap_mode="r")
        if a.ndim == 3:   # (S,H,W)
            return int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
        if a.ndim == 2:   # (H,W)
            return 1, int(a.shape[0]), int(a.shape[1])
    except Exception:
        pass
    return None

def _resize_nn(img: np.ndarray, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
    """
    最近邻重采样；默认 target_size=None（不缩放）。不改变亮度、对比度、分布。
    """
    if target_size is None:
        return img
    th, tw = target_size
    h, w = img.shape[:2]
    if (h, w) == (th, tw):
        return img
    # OpenCV 优先
    try:
        import cv2
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_NEAREST)
    except Exception:
        pass
    # numpy 降级
    rr = (np.linspace(0, h - 1, th)).astype(np.int64)
    cc = (np.linspace(0, w - 1, tw)).astype(np.int64)
    return img[rr[:, None], cc[None, :]]


def _to_2d(arr: np.ndarray) -> np.ndarray:
    """
    将输入规整为 2D (H, W)。
    允许输入形态： (H,W), (1,H,W), (H,W,1)
    其它形态回退取第 0 通道或安全零张量。
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        # (1,H,W) 或 (H,W,1) 或 (C,H,W)/(H,W,C)
        if a.shape[0] == 1 and a.ndim == 3:
            return a[0]
        if a.shape[-1] == 1:
            return a[..., 0]
        # 多通道：取第 0 通道（医疗场景通常为单通道）
        return a[..., 0] if a.shape[-1] <= a.shape[0] else a[0]
    # 其它异常形态：返回空占位
    return np.zeros((224, 224), dtype=np.float32)


# ============================== 数据集主体 ==============================

class ABMILDataset(Dataset):
    """
    纯净 ABMIL 数据集：不进行随机采样/归一化/增强，仅做 dtype 转换与可选尺寸重采样。
    - df 至少包含：id_column（默认 'checkno'）与多标签列（label_cols）
    - 输出：bags: (m, 1, H, W), labels: (C,)
    """

    def __init__(
        self,
        data_dir: str,
        df: pd.DataFrame,
        label_cols: List[str],
        bag_size: int = 10,

    # —— 以下为已存在参数，不需要更改 —— #
        slice_sampling: str = "center",
        seed: int = 42,
        normalize: str = "zscore",
        target_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
        id_column: str = "checkno",
        patient_file_suffix: str = "_img.npy",
        verbose: bool = True,

    # —— 新增 —— #
        return_meta: bool = False,  # 新增的参数
    ) -> None:
        self.return_meta = bool(return_meta)
        self.verbose = bool(verbose)
        self.slice_sampling = slice_sampling
        self.seed = seed
        self.normalize = normalize
        self.target_size = target_size
        self.augment = augment
        self.id_column = id_column
        self.patient_file_suffix = patient_file_suffix

        super().__init__()

        # 基本检查
        assert isinstance(data_dir, str) and os.path.isdir(data_dir), f"data_dir 不存在：{data_dir}"
        assert isinstance(df, pd.DataFrame), "df 必须为 pandas.DataFrame"
        assert id_column in df.columns, f"缺少列：{id_column}"
        assert isinstance(label_cols, (list, tuple)) and all(isinstance(c, str) for c in label_cols), "label_cols 必须为 str 列表"

        self.data_dir = data_dir
        self.df = df.reset_index(drop=True).copy()
        self.label_cols = list(label_cols)
        self.bag_size = int(bag_size)
        self.id_column = id_column
        self.patient_file_suffix = patient_file_suffix
        self.target_size = target_size

        # 一次性兼容警告
        self._warn_once = True
        self._compat_msg = (
            "[ABMILDataset] 本版已禁用：随机采样/归一化/增强；"
            "slice_sampling/normalize/augment 参数被忽略；仅中心窗口取片与可选尺寸重采样。"
        )

        # 盘点样本存储形态
        self._inv: Dict[str, Dict[str, Any]] = {}   # pid -> {mode, path, files, nslices}
        self._build_inventory()

        if verbose:
            print(
                f"[ABMILDataset] N={len(self.df)} | labels={self.label_cols} | "
                f"bag_size={self.bag_size} | id_col={self.id_column} | "
                f"target_size={self.target_size} | storage=single/file-folder"
            )

    # --------------------- 预索引：确定每个样本的文件来源 ---------------------

    def _build_inventory(self) -> None:
        """
        为每个样本建立索引：
        mode="direct"  -> 由 csv 的 npy_path 直读一个 .npy (S,H,W)
        mode="single"  -> data_dir/{pid}{self.patient_file_suffix}
        mode="folder"  -> data_dir/{pid}/*.npy
        mode="none"    -> 未找到
        """
        self._inv = {}

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            pid = str(row[self.id_column]).strip()

            # 0) 直读：csv 中提供了 npy_path
            if "npy_path" in self.df.columns:
                npy_path = str(row.get("npy_path", "")).strip()
                if npy_path and os.path.isfile(npy_path):
                    shp = _probe_npy_shape(npy_path)          # (S,H,W) 或 None
                    ns  = int(shp[0]) if shp else 0
                    self._inv[pid] = dict(
                        mode="direct",
                        path=npy_path,
                        files=None,
                        nslices=ns,
                    )
                    continue  # 命中直读后跳过后续 single/folder

            # A) 单文件 {pid}_img.npy
            p_single = os.path.join(self.data_dir, f"{pid}{self.patient_file_suffix}")
            if os.path.isfile(p_single):
                arr = _safe_load_npy(p_single)
                if arr is not None:
                    if arr.ndim == 2:
                        ns = 1
                    elif arr.ndim == 3:
                        ns = int(arr.shape[0])
                    else:
                        arr = np.squeeze(arr)
                        ns = 1 if arr.ndim == 2 else int(arr.shape[0])
                    self._inv[pid] = dict(mode="single", path=p_single, files=None, nslices=ns)
                    continue

            # B) 文件夹 {pid}/*.npy
            folder = os.path.join(self.data_dir, pid)
            if os.path.isdir(folder):
                files = sorted(glob.glob(os.path.join(folder, "*.npy")))
                if len(files) > 0:
                    self._inv[pid] = dict(mode="folder", path=folder, files=files, nslices=len(files))
                    continue

            # 未找到：记录空
            self._inv[pid] = dict(mode="none", path="", files=[], nslices=0)


    # --------------------- 索引策略：仅中心窗口 ---------------------

    @staticmethod
    def _center_indices(n: int, m: int) -> List[int]:
        """
        在 [0..n-1] 中取中心连续 m 个索引；若 n < m，按顺序循环补齐到 m。
        """
        if n <= 0:
            return []
        if n >= m:
            start = max((n - m) // 2, 0)
            return list(range(start, start + m))
        base = list(range(n))
        ext = []
        while len(base) + len(ext) < m:
            ext.extend(base)
        return (base + ext)[:m]

    # --------------------- 读取病例：单文件模式 ---------------------

    def _load_single(self, path: str, sel_idx: List[int]) -> np.ndarray:
        """
        读取单一 patient 级数组，返回形状 (m, 1, H, W); 不做归一化/增强。
        """
        # 1.3 新增：调试输出，确认读到的文件路径
        if self.verbose:
            print(f"[ABMILDataset] Loading file: {path}")

        arr = _safe_load_npy(path)
        if arr is None:
            H = self.target_size[0] if self.target_size else 224
            W = self.target_size[1] if self.target_size else 224
            return np.zeros((self.bag_size, 1, H, W), dtype=np.float32)

        a = np.squeeze(arr)
        if a.ndim == 2:
            a = a[None, ...]
        elif a.ndim != 3:
            # 无法解释的形态，回退零张量
            H = self.target_size[0] if self.target_size else 224
            W = self.target_size[1] if self.target_size else 224
            return np.zeros((self.bag_size, 1, H, W), dtype=np.float32)

        S, H, W = a.shape
        sel = [min(max(0, i), S - 1) for i in sel_idx]
        slides = []
        for i in sel:
            sl = _to_2d(a[i]).astype(np.float32, copy=False)
            sl = _resize_nn(sl, self.target_size)
            slides.append(sl[None, ...])  # (1,H,W)
        bag = np.stack(slides, axis=0).astype(np.float32, copy=False)  # (m,1,H,W)
        return bag
    def _load_direct(self, path: str, sel_idx: List[int]) -> np.ndarray:
        """
        读取病例：直读单个 .npy（与 single 模式完全相同）；
        返回形状 (m, 1, H, W)；不做归一化/增强。
        """
        if getattr(self, "verbose", False):
            print(f"[ABMILDataset] direct -> {path}")

        # 复用单文件读取逻辑，内部已处理( S,H,W / H,W / 1,H,W )以及缺失兜底
        return self._load_single(path, sel_idx)

    # --------------------- 读取病例：多文件模式 ---------------------

    def _load_folder(self, files: List[str], sel_idx: List[int]) -> np.ndarray:
        """
        读取多个切片文件，返回形状 (m, 1, H, W)；不做归一化/增强。
        """
        n = len(files)
        if n == 0:
            H = self.target_size[0] if self.target_size else 224
            W = self.target_size[1] if self.target_size else 224
            return np.zeros((self.bag_size, 1, H, W), dtype=np.float32)

        idxs = [min(max(0, i), n - 1) for i in sel_idx]
        chosen = [files[i] for i in idxs]
        slides = []
        for p in chosen:
            arr = _safe_load_npy(p)
            if arr is None:
                sl = np.zeros(self.target_size or (224, 224), dtype=np.float32)
            else:
                sl = _to_2d(arr).astype(np.float32, copy=False)
            sl = _resize_nn(sl, self.target_size)
            slides.append(sl[None, ...])  # (1,H,W)
        bag = np.stack(slides, axis=0).astype(np.float32, copy=False)  # (m,1,H,W)
        return bag

    # --------------------- PyTorch 标准接口 ---------------------
    def __getitem__(self, idx: int):
        """
        返回:
        - bag: (m, 1, H, W) float32
        - labels: (num_labels,) float32
        - (可选) meta: dict(checkno, slice_idx, seq_type)
        """
        row = self.df.iloc[idx]
        pid = str(row[self.id_column]).strip()
        info = self._inv.get(pid, {"mode": "none", "path": "", "files": [], "nslices": 0})
       # 期望 bag 大小与实际 n
        want = int(self.bag_size)
        n = int(info.get("nslices", 0))
        if n <= 0 and info["mode"] in ("direct", "single"):
            shp = _probe_npy_shape(info["path"])
            n = int(shp[0]) if shp else 0

        take = min(n, want)
        # 先做空包保护（一定要在 _center_indices 之前）
        if take <= 0:
            raise ValueError(
                    f"[ABMILDataset] No slices for checkno={pid}, path={info.get('path','')}"
                )
        # 再计算中心索引
        sel_idx = self._center_indices(n if n > 0 else take, take)
        # 读取/组装
        if info["mode"] == "direct":
            bag_np = self._load_direct(info["path"], sel_idx)
        elif info["mode"] == "single":
            bag_np = self._load_single(info["path"], sel_idx)
        elif info["mode"] == "folder":
            bag_np = self._load_folder(info["files"], sel_idx)
        else:
            H = self.target_size[0] if self.target_size else 224
            W = self.target_size[1] if self.target_size else 224
            bag_np = np.zeros((take, 1, H, W), dtype=np.float32)   # ← 这里用 take

        bags = torch.from_numpy(bag_np).float()
        labels = torch.as_tensor(row[self.label_cols].astype(float).values, dtype=torch.float32)

        if getattr(self, "return_meta", False):
            # —— 新增：优雅处理 seq_type 的 NaN/缺失 ——
            seq_val = row["seq_type"] if "seq_type" in self.df.columns else None
            try:
                import pandas as pd
                if pd.isna(seq_val):
                    seq_val = None
            except Exception:
                pass

            meta = {
                "checkno": pid,
                "slice_idx": torch.as_tensor(sel_idx, dtype=torch.long).view(-1),
                "seq_type": None if seq_val is None else str(seq_val),   # ← 这里替换你原来的那一行
                "bag_len": int(take),
            }

            if self.verbose and np.random.rand() < 0.01:  # 偶尔打印一条
                print(f"[DEBUG] checkno={pid} bag_len={int(take)} seq_type={meta['seq_type']}")
            return bags, labels, meta
        return bags, labels


    def __len__(self) -> int:
    # 基于 DataFrame 行数
        return int(len(self.df))

# ============================== 自检（本文件独立运行时） ==============================

if __name__ == "__main__":
    # 简单结构/路径自检；根据本地路径修改后执行
    fake = pd.DataFrame({
        "checkno": ["DEMO_A", "DEMO_B"],
        "L3/4": [0, 1],
        "L4/5": [1, 0],
        "L5/S1": [0, 1],
    })
    ds = ABMILDataset(
        data_dir="D:/RawMRI/Preprocessed",
        df=fake,
        label_cols=["L3/4", "L4/5", "L5/S1"],
        bag_size=10,
        slice_sampling="random",     # 将被忽略
        normalize="zscore",          # 将被忽略
        augment=True,                # 将被忽略
        target_size=None,            # 建议 None；如需统一尺寸可设 (H,W)
        verbose=True
    )
    x0, y0 = ds[0]
    print("bag:", tuple(x0.shape), "labels:", y0.numpy())
