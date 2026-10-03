"""optimizer.py — Bộ chọn tối ưu hoá, scheduler và cắt gradient (gradient clipping).

Công thức cần hiểu (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import torch
import torch.nn.utils as utils

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer theo tên cấu hình."""
    name_clean = name.lower()
    if name_clean not in OPTIMIZERS:
        raise ValueError(f"Optimizer '{name}' không nằm trong {OPTIMIZERS}")

    if name_clean == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif name_clean == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name_clean == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name_clean == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học, ví dụ cosine annealing."""
    if name is None:
        return None
    name_clean = name.lower()
    if name_clean == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    return None


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Khi dùng mixed precision FP16 + GradScaler: phải gọi scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    param_list = [p for p in params if p.grad is not None]
    if not param_list:
        return 0.0

    if max_norm is None or max_norm <= 0:
        total_norm = utils.clip_grad_norm_(param_list, max_norm=float("inf"))
    else:
        total_norm = utils.clip_grad_norm_(param_list, max_norm=float(max_norm))

    return float(total_norm)
