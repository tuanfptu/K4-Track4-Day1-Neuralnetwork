"""data.py — Nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    proc_path = Path(processed_dir)
    train_file = proc_path / "train.npz"
    eval_file = proc_path / "eval.npz"

    if not train_file.exists() or not eval_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file npz trong {processed_dir}. Hãy chạy scripts/split_data.py trước.")

    train_data = np.load(train_file)
    eval_data = np.load(eval_file)

    X_train_full = train_data["X"].astype(np.float32)
    y_train_full = train_data["y"].astype(np.int64)

    X_eval = eval_data["X"].astype(np.float32)
    y_eval = eval_data["y"].astype(np.int64)
    eval_row_id = eval_data["row_id"].astype(np.int64)

    assert X_train_full.ndim == 2 and X_train_full.shape[1] == 54, f"X_train_full shape sai: {X_train_full.shape}"
    assert y_train_full.ndim == 1 and len(y_train_full) == len(X_train_full), "y_train_full shape sai"
    assert X_eval.shape[1] == 54 and len(y_eval) == len(X_eval), "X_eval / y_eval shape sai"
    assert len(eval_row_id) == len(X_eval), "eval_row_id length sai"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    """
    num_part = X_tr[:, :N_NUMERIC]
    mean = num_part.mean(axis=0)
    std = num_part.std(axis=0)
    # Tránh chia cho 0 nếu một cột có độ lệch chuẩn = 0
    std = np.where(std == 0.0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_scaled = np.array(X, dtype=np.float32, copy=True)
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / std
    return X_scaled


def prepare_data(device: str | torch.device, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id, mean, std
    """
    dev = torch.device(device)
    X_train_full, y_train_full, X_eval_raw, y_eval_raw, eval_row_id = load_split(processed_dir)

    # 1. Tách val từ train
    X_tr_raw, y_tr_raw, X_val_raw, y_val_raw = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )

    # 2. Tính standardizer CHỈ trên X_tr
    mean, std = fit_standardizer(X_tr_raw)

    # 3. Áp dụng chuẩn hoá cho cả 3 tập
    X_tr_std = apply_standardizer(X_tr_raw, mean, std)
    X_val_std = apply_standardizer(X_val_raw, mean, std)
    X_eval_std = apply_standardizer(X_eval_raw, mean, std)

    # 4. Chuyển lên device
    X_tr = torch.as_tensor(X_tr_std, dtype=torch.float32, device=dev)
    y_tr = torch.as_tensor(y_tr_raw, dtype=torch.int64, device=dev)

    X_val = torch.as_tensor(X_val_std, dtype=torch.float32, device=dev)
    y_val = torch.as_tensor(y_val_raw, dtype=torch.int64, device=dev)

    X_eval = torch.as_tensor(X_eval_std, dtype=torch.float32, device=dev)
    y_eval = torch.as_tensor(y_eval_raw, dtype=torch.int64, device=dev)

    # 5. In thống kê
    ctr = np.bincount(y_tr_raw, minlength=7)
    cval = np.bincount(y_val_raw, minlength=7)
    majority_class = int(ctr.argmax())
    majority_acc = float(cval[majority_class] / len(y_val_raw))

    print(f"Data prepared on device '{dev}':")
    print(f"  Train: X={X_tr.shape}, y={y_tr.shape}")
    print(f"  Val  : X={X_val.shape}, y={y_val.shape}")
    print(f"  Eval : X={X_eval.shape}, y={y_eval.shape}")
    print(f"  Majority class baseline on Val: class {majority_class} -> acc = {majority_acc:.4f}")

    return {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_val": X_val,
        "y_val": y_val,
        "X_eval": X_eval,
        "y_eval": y_eval,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
        "majority_acc": majority_acc,
    }


def iterate_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int,
                    generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader."""
    N = len(X)
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=X.device)
    else:
        perm = torch.arange(N, device=X.device)

    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
