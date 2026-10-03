"""results_table.py — Lưu kết quả thí nghiệm ra JSON và xuất bảng experiments.xlsx.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx.

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, không ghi đè)
"""
from __future__ import annotations

import json
from pathlib import Path
import openpyxl

EXPERIMENT_COLS = [
    "exp_id", "group", "description", "loss", "optimizer", "lr", "weight_decay",
    "batch", "epochs", "hidden", "dropout", "clip_norm", "precision", "init",
    "seed", "step0_loss", "best_val_loss", "best_epoch", "final_train_loss",
    "final_val_loss", "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB",
    "diverged", "eval_acc", "eval_macro_f1", "figure_file", "notes"
]


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result["cfg"], result["history"], result["summary"] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có.
    """
    p = Path(results_dir)
    p.mkdir(parents=True, exist_ok=True)
    exp_id = result["cfg"]["exp_id"]
    out_file = p / f"{exp_id}.json"

    data_to_save = {
        "cfg": result["cfg"],
        "history": result["history"],
        "summary": result["summary"],
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    return str(out_file)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    p = Path(results_dir)
    if not p.exists():
        return []

    results = []
    for f in sorted(p.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            results.append(json.load(fp))
    results.sort(key=lambda r: r.get("cfg", {}).get("exp_id", ""))
    return results


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file.
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng.
    """
    cfg = result["cfg"]
    summary = result.get("summary", {})
    exp_id = cfg.get("exp_id", "")

    # Format hidden tuple to string "256-128"
    hidden = cfg.get("hidden", (256, 128))
    hidden_str = "-".join(map(str, hidden)) if isinstance(hidden, (list, tuple)) else str(hidden)

    # Format clip_norm
    clip_val = cfg.get("clip_norm", None)
    clip_str = "none" if clip_val is None else clip_val

    row = {
        "exp_id": exp_id,
        "group": cfg.get("group", ""),
        "description": cfg.get("description", ""),
        "loss": cfg.get("loss", "").upper(),
        "optimizer": cfg.get("optimizer", ""),
        "lr": cfg.get("lr", None),
        "weight_decay": cfg.get("weight_decay", 0.0),
        "batch": cfg.get("batch", 512),
        "epochs": cfg.get("epochs", 20),
        "hidden": hidden_str,
        "dropout": cfg.get("dropout", 0.0),
        "clip_norm": clip_str,
        "precision": cfg.get("precision", "fp32"),
        "init": cfg.get("init", "he"),
        "seed": cfg.get("seed", 1),
        "step0_loss": summary.get("step0_loss", None),
        "best_val_loss": summary.get("best_val_loss", None),
        "best_epoch": summary.get("best_epoch", None),
        "final_train_loss": summary.get("final_train_loss", None),
        "final_val_loss": summary.get("final_val_loss", None),
        "val_acc": summary.get("val_acc", None),
        "val_macro_f1": summary.get("val_macro_f1", None),
        "time_per_epoch_s": summary.get("time_per_epoch_s", None),
        "peak_mem_MB": summary.get("peak_mem_MB", None),
        "diverged": summary.get("diverged", False),
        "eval_acc": eval_scores.get("accuracy", None) if eval_scores else None,
        "eval_macro_f1": eval_scores.get("macro_f1", None) if eval_scores else None,
        "figure_file": f"figures/{exp_id}.png",
        "notes": notes or cfg.get("notes", ""),
    }
    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path.

    Các bước:
      1. wb = openpyxl.load_workbook(template_path)   # KHÔNG dùng data_only=True
      2. ws = wb["Experiments"]; đọc tiêu đề dòng 1 để biết cột nào ứng với khoá nào
      3. với mỗi row: ghi giá trị vào đúng cột; BỎ QUA các cột công thức
      4. wb.save(out_path)
    """
    wb = openpyxl.load_workbook(template_path)
    ws = wb["Experiments"]

    # Đọc headers dòng 1
    col_mapping = {}
    for col_idx in range(1, 35):
        val = ws.cell(1, col_idx).value
        if val:
            col_mapping[val] = col_idx

    # Ghi dữ liệu từng dòng
    for i, row in enumerate(rows):
        r_idx = i + 2
        for key, val in row.items():
            if key in col_mapping:
                col_num = col_mapping[key]
                ws.cell(r_idx, col_num, val)

        # Đảm bảo các cột công thức có ở dòng r_idx
        ws.cell(r_idx, 30, f'=IF(P{r_idx}="","",P{r_idx}-LN(7))')
        ws.cell(r_idx, 31, f'=IF(OR(T{r_idx}="",S{r_idx}=""),"",T{r_idx}-S{r_idx})')
        ws.cell(r_idx, 32, f'=IF(OR(V{r_idx}="",Seeds!$C$8=""),"",V{r_idx}-Seeds!$C$8)')
        ws.cell(r_idx, 33, f'=IF(OR(AF{r_idx}="",Seeds!$C$10=""),"",IF(ABS(AF{r_idx})>Seeds!$C$10,"Có","Không"))')

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
