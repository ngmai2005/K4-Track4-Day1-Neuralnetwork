"""plots.py — Vẽ biểu đồ huấn luyện từng thí nghiệm và biểu đồ so sánh nhóm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    """
    cfg = result["cfg"]
    hist = result["history"]
    summary = result.get("summary", {})
    epochs = hist["epoch"]
    best_epoch = summary.get("best_epoch", None)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Cấu hình chuỗi tiêu đề
    title_info = (f"[{cfg.get('exp_id', '')}] {cfg.get('optimizer', '')} | "
                  f"lr={cfg.get('lr', '')} | b={cfg.get('batch', '')} | "
                  f"init={cfg.get('init', '')} | drop={cfg.get('dropout', 0.0)}")
    fig.suptitle(title_info, fontsize=12, fontweight="bold")

    # Ô 1: Loss
    ax0 = axes[0]
    ax0.plot(epochs, hist["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax0.plot(epochs, hist["val_loss"], label="Val Loss", color="#d62728", lw=2)
    if best_epoch is not None and best_epoch in epochs:
        ax0.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7,
                    label=f"Best Ep ({best_epoch})")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.set_title("Train & Val Loss")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # Ô 2: Metrics (Accuracy & Macro-F1)
    ax1 = axes[1]
    ax1.plot(epochs, hist["val_acc"], label="Val Acc", color="#2ca02c", lw=2)
    if "val_macro_f1" in hist and len(hist["val_macro_f1"]) == len(epochs):
        ax1.plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="#ff7f0e", lw=2)
    if best_epoch is not None and best_epoch in epochs:
        ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Score")
    ax1.set_title("Validation Accuracy & Macro-F1")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Ô 3: Gradient Norm
    ax2 = axes[2]
    ax2.plot(epochs, hist["grad_norm"], label="Grad Norm (pre-clip)", color="#9467bd", lw=2)
    if cfg.get("clip_norm") is not None:
        ax2.axhline(float(cfg["clip_norm"]), color="red", linestyle=":", label=f"Clip c={cfg['clip_norm']}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("L2 Norm")
    ax2.set_title("Average Gradient Norm")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số ("val_loss", "val_macro_f1", "grad_norm", ...) của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for r in results:
        cfg = r["cfg"]
        hist = r["history"]
        exp_id = cfg.get("exp_id", "unnamed")
        if metric in hist:
            ax.plot(hist["epoch"], hist[metric], label=exp_id, lw=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(title or f"So sánh {metric} giữa các thí nghiệm", fontsize=12, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
