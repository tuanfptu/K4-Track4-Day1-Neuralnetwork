"""train.py — Pipeline huấn luyện, đánh giá, kiểm tra sức khoẻ mô hình và ghi dự đoán.

Mọi thí nghiệm chỉ là đổi dict cfg rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # tìm trên tập val
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc float
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits ở eval mode."""
    model.eval()
    preds = []
    N = len(X)
    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=1))
    return torch.cat(preds, dim=0)


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (lấy mean trên toàn bộ phần tử).
    """
    loss_name_clean = loss_name.lower()
    if loss_name_clean == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name_clean == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        return F.mse_loss(logits, y_onehot, reduction="mean")
    else:
        raise ValueError(f"Hàm mất mát không được hỗ trợ: {loss_name}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1, confusion) ở chế độ eval() (dropout tắt) và no_grad."""
    model.eval()
    N = len(X)
    total_loss = 0.0
    all_preds = []

    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name)
        total_loss += float(loss.item()) * len(xb)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = y.cpu().numpy()

    acc = float((y_pred == y_true).mean())

    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    macro_f1 = macro_f1_from_confusion(cm)

    return {
        "loss": float(total_loss / N),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "confusion": cm,
    }


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    set_seed(cfg["seed"])
    dev = data["X_tr"].device
    is_cuda = (dev.type == "cuda")

    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = cfg.get("init", "he")

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(dev)
    assert count_params(model) == EXPECTED_PARAMS[hidden], (
        f"Số tham số {count_params(model)} không khớp {EXPECTED_PARAMS[hidden]}"
    )

    optimizer = build_optimizer(
        cfg["optimizer"], model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
        momentum=cfg.get("momentum", 0.9),
    )

    # Cấu hình Mixed Precision
    precision = cfg.get("precision", "fp32").lower()
    use_amp = False
    amp_dtype = torch.float32
    scaler = None

    if is_cuda and precision in ("fp16", "bf16"):
        use_amp = True
        if precision == "fp16":
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")
        elif precision == "bf16":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                # GPU không hỗ trợ bfloat16 thì fallback fp32
                use_amp = False

    # 1. Loss bước 0 trên tập val (eval mode, trước khi update bất kỳ bước nào)
    step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = step0_res["loss"]

    # Đặt generator cho shuffle batch
    if is_cuda:
        gen = torch.Generator(device=dev).manual_seed(cfg["seed"])
    else:
        gen = torch.Generator().manual_seed(cfg["seed"])

    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch"])
    clip_norm = cfg.get("clip_norm")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    best_state = None
    diverged = False

    # Để đánh giá train loss nhanh mỗi epoch, dùng tập con cố định 50.000 mẫu train (theo gợi ý GUIDE)
    sub_size = min(50_000, len(data["X_tr"]))
    X_tr_eval_sub = data["X_tr"][:sub_size]
    y_tr_eval_sub = data["y_tr"][:sub_size]

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(dev)

    for epoch in range(1, epochs + 1):
        if is_cuda:
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        model.train()
        grad_norms_epoch = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size, generator=gen, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                print(f"[{cfg['exp_id']}] Diverged at epoch {epoch} (loss = {loss.item()})")
                break

            if scaler is not None:
                scaler.scale(loss).backward()
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            grad_norms_epoch.append(gn)

        if diverged:
            break

        if is_cuda:
            torch.cuda.synchronize()
        t_epoch = time.perf_counter() - t_start

        avg_gn = float(np.mean(grad_norms_epoch)) if grad_norms_epoch else 0.0

        # Đánh giá cuối epoch ở chế độ eval
        tr_res = evaluate(model, X_tr_eval_sub, y_tr_eval_sub, loss_name=cfg["loss"])
        val_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_acc"].append(val_res["acc"])
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["grad_norm"].append(avg_gn)
        history["epoch_time_s"].append(t_epoch)

        # Lưu best_state theo best val loss
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            best_val_acc = val_res["acc"]
            best_val_macro_f1 = val_res["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Tổng kết
    final_tr_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else float("nan")
    avg_epoch_time = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0

    peak_mem_MB = 0.0
    if is_cuda:
        peak_mem_MB = float(torch.cuda.max_memory_allocated(dev) / (1024 * 1024))

    summary = {
        "exp_id": cfg["exp_id"],
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(final_tr_loss),
        "final_val_loss": float(final_val_loss),
        "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_macro_f1),
        "time_per_epoch_s": float(avg_epoch_time),
        "peak_mem_MB": float(peak_mem_MB),
        "diverged": diverged,
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề row_id,pred."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"row_id": row_id.astype(int), "pred": preds.astype(int)})
    df.to_csv(out_path, index=False)
    print(f"Ghi file dự đoán ({len(df)} dòng) -> {out_path}")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> np.ndarray:
    """Dùng cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi CSV."""
    dev = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = cfg.get("init", "he")

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(dev)
    if result["best_state"] is not None:
        model.load_state_dict({k: v.to(dev) for k, v in result["best_state"].items()})
    model.eval()

    preds = predict(model, data["X_eval"]).cpu().numpy()
    write_predictions(data["eval_row_id"], preds, pred_path)
    return preds
