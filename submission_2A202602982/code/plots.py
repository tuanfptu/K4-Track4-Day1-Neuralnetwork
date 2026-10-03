"""plots.py — Vẽ biểu đồ cho từng thí nghiệm và biểu đồ so sánh chồng theo nhóm.

Mỗi thí nghiệm bắt buộc có một ảnh figures/<exp_id>.png.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = result["cfg"]
    hist = result["history"]
    summary = result.get("summary", {})
    best_epoch = summary.get("best_epoch", -1)

    epochs = hist["epoch"]
    if not epochs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Loss
    ax = axes[0]
    ax.plot(epochs, hist["train_loss"], label="Train Loss (sub-eval)", color="tab:blue", marker="o", markersize=3)
    ax.plot(epochs, hist["val_loss"], label="Val Loss", color="tab:orange", marker="s", markersize=3)
    if best_epoch in epochs:
        ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Ep ({best_epoch})")
    ax.set_title("Loss vs. Epoch", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 2. Accuracy & Macro-F1
    ax = axes[1]
    ax.plot(epochs, hist["val_acc"], label="Val Acc", color="tab:green", marker="^", markersize=3)
    ax.plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="tab:purple", marker="d", markersize=3)
    if best_epoch in epochs:
        ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Ep ({best_epoch})")
    ax.set_title("Val Accuracy & Macro-F1", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric Score")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 3. Gradient Norm
    ax = axes[2]
    ax.plot(epochs, hist["grad_norm"], label="Grad Norm (pre-clip)", color="tab:red", marker="x", markersize=3)
    ax.set_title("Average Gradient Norm", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Norm L2")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    title_str = (
        f"[{cfg.get('exp_id', 'exp')}] {cfg.get('description', '')} | "
        f"opt={cfg.get('optimizer')}, lr={cfg.get('lr')}, batch={cfg.get('batch')}, init={cfg.get('init')}"
    )
    fig.suptitle(title_str, fontsize=12, fontweight="bold", y=1.03)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ 'val_loss', 'val_macro_f1', 'grad_norm') của nhiều thí nghiệm
    trên cùng một trục để so sánh trực tiếp.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for res in results:
        cfg = res["cfg"]
        hist = res["history"]
        exp_id = cfg.get("exp_id", "exp")
        epochs = hist.get("epoch", [])
        vals = hist.get(metric, [])
        if epochs and vals:
            ax.plot(epochs, vals, marker="o", markersize=3, label=f"{exp_id} ({cfg.get('description', '')})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(title or f"So sánh {metric}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
