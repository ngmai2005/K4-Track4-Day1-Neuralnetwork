"""train.py — Huấn luyện mô hình, đánh giá, ghi nhật ký và tạo dự đoán eval.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # Chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    Đúng công thức chuẩn hóa trong scripts/evaluate.py.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô; gom argmax(dim=1); torch.cat.
    """
    model.eval()
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1))
    return torch.cat(preds, dim=0)


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (lấy trung bình trên mọi phần tử theo nn.MSELoss).
    """
    if loss_name.lower() == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name.lower() == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.shape[-1]).float()
        return F.mse_loss(logits, y_onehot)
    else:
        raise ValueError(f"Hàm mất mát không được hỗ trợ: {loss_name}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    """
    model.eval()
    n_samples = len(X)
    total_loss = 0.0
    preds_list = []

    for i in range(0, n_samples, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name)
        total_loss += float(loss.item()) * len(xb)
        preds_list.append(torch.argmax(logits, dim=1))

    mean_loss = total_loss / n_samples
    preds = torch.cat(preds_list, dim=0)
    acc = float((preds == y).float().mean().item())

    y_np = y.cpu().numpy()
    pred_np = preds.cpu().numpy()
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (y_np, pred_np), 1)
    macro_f1 = macro_f1_from_confusion(cm)

    return {
        "loss": float(mean_loss),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "cm": cm,
    }


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất}
    """
    set_seed(cfg["seed"])
    device = data["X_tr"].device

    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = str(cfg.get("init", "he"))

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden], (
        f"Số tham số không khớp: {count_params(model)} != {EXPECTED_PARAMS[hidden]}"
    )

    optimizer = build_optimizer(
        cfg["optimizer"],
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        momentum=float(cfg.get("momentum", 0.9)),
    )

    precision = cfg.get("precision", "fp32")
    scaler = None
    if precision == "fp16" and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    # Đo loss bước 0 trên val ở chế độ eval() trước cập nhật đầu tiên
    step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = step0_res["loss"]

    # Tập con cố định 50 000 mẫu train để đo train_loss ở eval() mode chuẩn mực và nhanh
    sub_size = min(50_000, len(data["X_tr"]))
    X_tr_eval = data["X_tr"][:sub_size]
    y_tr_eval = data["y_tr"][:sub_size]

    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch", 512))
    clip_norm = cfg.get("clip_norm", None)

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
    best_epoch = 1
    best_state = None
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    diverged = False

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        grad_norms_epoch = []

        generator = torch.Generator(device=device).manual_seed(cfg["seed"] * 1000 + epoch)

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size,
                                      generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if precision in ("fp16", "bf16") and device.type == "cuda":
                amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
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

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        if diverged:
            print(f"[{cfg.get('exp_id')}] Phan ky tai epoch {epoch}!")
            break

        # Đánh giá cuối epoch ở chế độ eval()
        train_res = evaluate(model, X_tr_eval, y_tr_eval, loss_name=cfg["loss"])
        val_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])

        avg_gn = float(np.mean(grad_norms_epoch)) if grad_norms_epoch else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_res["loss"], 4))
        history["val_loss"].append(round(val_res["loss"], 4))
        history["val_acc"].append(round(val_res["acc"], 4))
        history["val_macro_f1"].append(round(val_res["macro_f1"], 4))
        history["grad_norm"].append(round(avg_gn, 4))
        history["epoch_time_s"].append(round(epoch_time, 2))

        # Lưu best_state tại epoch có val_loss thấp nhất
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_acc = val_res["acc"]
            best_val_macro_f1 = val_res["macro_f1"]

    peak_mem_MB = 0.0
    if device.type == "cuda":
        peak_mem_MB = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    mean_epoch_time = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0

    summary = {
        "step0_loss": round(float(step0_loss), 4),
        "best_val_loss": round(float(best_val_loss), 4) if not diverged else None,
        "best_epoch": int(best_epoch) if not diverged else None,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "val_acc": round(float(best_val_acc), 4) if not diverged else None,
        "val_macro_f1": round(float(best_val_macro_f1), 4) if not diverged else None,
        "time_per_epoch_s": round(mean_epoch_time, 2),
        "peak_mem_MB": round(peak_mem_MB, 2),
        "diverged": diverged,
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    df = pd.DataFrame({"row_id": row_id, "pred": preds})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Ghi du doan vao: {path} ({len(df)} dong)")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    """
    device = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = str(cfg.get("init", "he"))

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)
    model.load_state_dict(result["best_state"])
    model.eval()

    preds = predict(model, data["X_eval"])
    preds_np = preds.cpu().numpy()
    write_predictions(data["eval_row_id"], preds_np, pred_path)
