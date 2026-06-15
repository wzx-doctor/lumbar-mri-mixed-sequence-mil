# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# -------------------------------------------------------
# Encoder（沿用并保持与当前工程一致）
# -------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 兼容外层 DataLoader 产生的 5D 输入 [1, m, 1, H, W]
        if x.dim() == 5 and x.size(0) == 1:
            x = x.squeeze(0)  # (m, 1, H, W)
        h = self.conv(x).flatten(1)      # (m, 128)
        h = self.fc(h)                   # (m, out_dim)
        return F.relu(h, inplace=True)   # 保持你当前风格（与现版一致）  # :contentReference[oaicite:1]{index=1}


# -------------------------------------------------------
# Gated + Learnable-Temperature Attention（内部单元）
# -------------------------------------------------------
# 顶部已导入: import torch.nn.functional as F   （文件里已有）
class _GatedTempAttn(nn.Module):
    """
    Gated + Temperature Attention 单元
    输入:  H ∈ R^{m, d}  (m: 切片数, d: 特征维)
    输出:  z ∈ R^{d}    (bag-level 表征, 按 w 加权平均)
          w ∈ R^{m}    (每个切片的注意力权重, softmax 后和为 1)
    """
    def __init__(
        self,
        in_dim: int,
        attn_dim: int = 128,
        init_tau: float = 0.7,
        dropout: float = 0.1,          # 对中间门控特征 A 的 dropout
        attn_dropout_p: float = 0.3    # 对 attention logits 的 dropout（softmax 前），训练态生效
    ):
        super().__init__()
        # 门控注意力：A = tanh(Wa H) ⊙ sigmoid(Wb H)
        self.Wa   = nn.Linear(in_dim, attn_dim, bias=True)
        self.Wb   = nn.Linear(in_dim, attn_dim, bias=True)
        self.v    = nn.Linear(attn_dim, 1, bias=False)
        self.norm = nn.LayerNorm(attn_dim)
        self.drop = nn.Dropout(p=dropout)

        # logits dropout 概率
        self.attn_dropout_p = float(attn_dropout_p) if attn_dropout_p is not None else 0.0

        # 温度参数（对 softmax 进行温度缩放）
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(init_tau))))

    @property
    def temperature(self) -> float:
        """返回当前温度（clamp 到 [0.2, 5.0]），便于监控日志/诊断。"""
        with torch.no_grad():
            return float(torch.clamp(self.log_tau.exp(), 0.2, 5.0).cpu().item())

    def forward(self, H: torch.Tensor):
        """
        H: (m, d)
        返回:
            z: (d,) — bag 级表征（按 w 加权平均后的特征）
            w: (m,) — 注意力权重（softmax 归一化）
        """
        # 门控 + 规范化 + dropout
        A1 = torch.tanh(self.Wa(H))           # (m, attn_dim)
        A2 = torch.sigmoid(self.Wb(H))        # (m, attn_dim)
        A  = self.drop(self.norm(A1 * A2))    # (m, attn_dim)

        # attention logits
        s = self.v(A).squeeze(-1)             # (m,)

        # 训练时对 logits 做 dropout（有助于抑制过拟合）
        if self.attn_dropout_p > 0 and self.training:
            s = F.dropout(s, p=self.attn_dropout_p)

        # 温度缩放
        tau = torch.clamp(self.log_tau.exp(), 0.2, 5.0)

        # 注意力权重
        w = torch.softmax(s / tau, dim=0)     # (m,)

        # bag 特征：按注意力权重加权平均
        z = torch.sum(w.unsqueeze(-1) * H, dim=0)  # (d,)

        # 关键：返回 (z, w) 以便外部可视化/导出
        return z, w

# -------------------------------------------------------
# 新增：顶层 GatedTempAttentionMIL（满足 train_abmil.py 要求）
# -------------------------------------------------------
class GatedTempAttentionMIL(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        attn_dim: int = 128,
        num_classes: int = 3,
        init_tau: float = 0.7,
        dropout: float = 0.1,              # 对 A 的 dropout（保留）
        attn_dropout: float = 0.3,         # ← 新增：对 logits 的 dropout
        return_attention: bool = False
    ):
        super().__init__()
        self.encoder = SmallCNN(out_dim=input_dim)
        self.attn = _GatedTempAttn(
            input_dim, attn_dim=attn_dim,
            init_tau=init_tau,
            dropout=dropout,
            attn_dropout_p=attn_dropout      # ← 往下传
        )
        self.classifier = nn.Linear(input_dim, num_classes)
        self.return_attention = return_attention

    def forward(self, bag: torch.Tensor, return_attention: bool = None):
        # 兼容 (1, m, 1, H, W) / (m, 1, H, W)
        if bag.dim() == 5 and bag.size(0) == 1:
            bag = bag.squeeze(0)
        H = self.encoder(bag)           # (m, d)
        z, w = self.attn(H)             # (d,), (m,)
        logits = self.classifier(z)     # (C,)
        want_attn = self.return_attention if return_attention is None else bool(return_attention)
        if want_attn:
            return logits, w, H
        return logits

    # 提供温度读取接口（便于日志/诊断）
    @property
    def temperature(self) -> float:
        return float(torch.clamp(self.attn.log_tau.exp(), 0.2, 5.0).detach().cpu().item())


# -------------------------------------------------------
# ABMILWithCNN（保留，并扩展返回以适配诊断链路）
# -------------------------------------------------------
class ABMILWithCNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 3,
        init_tau: float = 0.7,
        # 这里新增 A 侧的 dropout（softmax 前的 A）
        a_dropout: Optional[float] = None,
        # 这里是 logits 的 dropout（softmax 前的 s）
        attn_dropout: float = 0.3,
        return_attention: bool = False,
    ):
        super().__init__()

        # 如果 a_dropout 没给，则回退到 attn_dropout（保持你现在的 CLI 习惯）
        self.a_dropout = a_dropout if a_dropout is not None else attn_dropout
        self.attn_dropout = attn_dropout

        self.encoder = SmallCNN(out_dim=hidden_dim)

        # 注意：把 A 的 dropout 和 logits 的 dropout 都传给注意力模块
        self.attn = _GatedTempAttn(
            in_dim=hidden_dim,
            attn_dim=hidden_dim,
            init_tau=init_tau,
            dropout=self.a_dropout,               # ← A 的 dropout
            attn_dropout_p=self.attn_dropout,     # ← logits 的 dropout
        )

        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.return_attention = return_attention
    
    def forward(self, bag: torch.Tensor, return_attention: bool = None):
        # 兼容 (1, m, 1, H, W) / (m, 1, H, W)
        if bag.dim() == 5 and bag.size(0) == 1:
            bag = bag.squeeze(0)

        H = self.encoder(bag)       # (m, d)
        z, w = self.attn(H)         # (d,), (m,)
        logits = self.classifier(z) # (C,)

        want_attn = self.return_attention if return_attention is None else bool(return_attention)
        if want_attn:
            return logits, w, H
        return logits

# -------------------------------------------------------
# 其余 MIL 基线（保持不变，便于消融与对照）
# -------------------------------------------------------
class MaxPoolingMIL(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=3):
        super().__init__()
        self.encoder = SmallCNN(out_dim=input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, bag: torch.Tensor):
        if bag.dim() == 5 and bag.size(0) == 1:
            bag = bag.squeeze(0)
        H = self.encoder(bag)           # (m, d)
        pooled, _ = torch.max(H, dim=0) # (d,)
        return self.fc(pooled)


class MeanPoolingMIL(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=3):
        super().__init__()
        self.encoder = SmallCNN(out_dim=input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, bag: torch.Tensor):
        if bag.dim() == 5 and bag.size(0) == 1:
            bag = bag.squeeze(0)
        H = self.encoder(bag)           # (m, d)
        pooled = torch.mean(H, dim=0)   # (d,)
        return self.fc(pooled)
