# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Unified detection evaluation — supports both COCO JSON and YOLO-format GT.

Mirrors Tridos vid_infer_eval.py: COCO eval + custom Pd/Fa/mAP/PR curves.

Usage:
    # COCO-format GT (IRDST, ITSDT)
    python eval/eval_det.py --gt-format coco --coco-gt .../instances_test2017.json ...

    # YOLO-format GT (DAUB, MIST_det)
    python eval/eval_det.py --gt-format yolo --data .../DAUB_YOLO/dataset.yaml ...

Stages:
  1. YOLO inference → predictions + (可选) vis/<seq>/<img>.png
  2. COCO/pycocotools eval + custom Pd/Fa metrics
     → metrics.json   • COCO 统计置于顶层，键名与 51_TDC/TDCNet 一致
     → pr_curve.json  • pycocotools 官方 101 点插值曲线（schema_version 2）
     → summary.csv    • TDCNet 口径的 COCO 统计；另出 summary_custom.csv（Pd/Fa 口径）
     → error_check/<seq>.xlsx • 逐序列漏检/误检（同 TDCNet）
     → gt_coco.json   • 仅 --gt-format yolo 时额外导出，供 51_TDC/tools/eval_diag.py 使用

与 51_TDC/TDCNet 的口径对齐：
  * metrics.json 顶层键：dataset / model_path / images / AP / AP50 / AP75 /
    AP_small / AP_medium / AP_large / AR_1 / AR_10 / AR_100 /
    precision50_coco / recall50_coco / f1_50_coco
  * precision50_coco / recall50_coco / f1_50_coco 取自 COCOeval 官方 PR 曲线
    （与 TDCNet vid_infer_eval.py 同源公式），不是自定义阈值指标
  * 评估范围默认覆盖完整 GT 图像列表（含无标注背景帧），与 TDCNet 一致；
    想要旧口径（只统计有预测的图）用 --eval-scope pred
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from ultralytics import YOLO

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    _HAS_PYCOCOTOOLS = True
except ImportError:
    _HAS_PYCOCOTOOLS = False

try:
    from scipy.io import savemat as _savemat

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COCO_IOU_THRESHOLDS = [round(float(x), 2) for x in np.arange(0.50, 0.96, 0.05)]
SUMMARY_FIELDS = [
    "dataset",
    "images",
    "gt",
    "predictions",
    "AP",
    "AP50",
    "AP75",
    "precision50",
    "recall50",
    "f1_50",
    "score50",
    "Pd",
    "Fa",
    "mean_iou50",
    "tp50",
    "fp50",
    "fn50",
]

# 与 51_TDC/TDCNet vid_infer_eval.py 完全一致的 summary.csv 列集合
COCO_SUMMARY_FIELDS = [
    "dataset",
    "images",
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "precision50_coco",
    "recall50_coco",
    "f1_50_coco",
]

_FONT_CANDIDATES = [
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "arial.ttf",
    "C:/Windows/Fonts/simhei.ttf",
]


# ===========================================================================
# Visualization helpers
# ===========================================================================
def _resolve_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        if os.path.isfile(candidate):
            try:
                return ImageFont.truetype(candidate, 13)
            except OSError:
                continue
    return ImageFont.load_default()


def extract_sequence_name(file_name: str) -> str:
    """从 file_name 推断序列名。.

    'images/62/0.bmp' -> '62' (COCO IRDST) 'images/test/data5/0.bmp' -> 'data5' (COCO DAUB) '62_0.bmp' -> '62' (YOLO
    扁平命名)
    """
    parts = Path(file_name).parts
    if len(parts) >= 2:
        return parts[-2]
    stem = Path(file_name).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return "default"


def draw_boxes_on_image(image, boxes_xyxy, scores, labels, class_names, colors, font, line_width=2):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for box, score, cls_id in zip(boxes_xyxy, scores, labels):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = max(0.0, min(float(image.width), x1))
        x2 = max(0.0, min(float(image.width), x2))
        y1 = max(0.0, min(float(image.height), y1))
        y2 = max(0.0, min(float(image.height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        color = colors[cls_id % len(colors)]
        name = class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}"
        label = f"{name} {score:.2f}"
        for t in range(max(1, line_width)):
            draw.rectangle([int(x1) + t, int(y1) + t, int(x2) - t, int(y2) - t], outline=color)
        try:
            bbox = draw.textbbox((int(x1), int(y1)), label, font=font)
        except (AttributeError, TypeError):
            bbox = [int(x1), int(y1) - 14, int(x1) + len(label) * 8, int(y1)]
        text_h, text_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        ty = max(0, int(y1) - text_h - 2)
        draw.rectangle([int(x1), ty, int(x1) + text_w + 4, ty + text_h + 2], fill=color)
        draw.text((int(x1) + 2, ty + 1), label, fill=(255, 255, 255), font=font)
    return image


# ===========================================================================
# Helpers
# ===========================================================================
def xywh_to_xyxy(box):
    x, y, w, h = [float(v) for v in box]
    return x, y, x + w, y + h


def bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else float(inter / denom)


def compute_ap(recalls, precisions):
    if recalls.size == 0:
        return 0.0
    r_points = np.linspace(0.0, 1.0, 101)
    return float(np.mean([float(precisions[recalls >= r].max()) if (recalls >= r).any() else 0.0 for r in r_points]))


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ===========================================================================
# TDCNet-compatible extras: official COCO PR curve / per-sequence error check
# ===========================================================================
def coco_pr_stats(coco_eval):
    """取 COCOeval 官方 PR 曲线上的 @IoU0.50 P/R/F1（与 TDCNet 同源公式）。."""
    precisions_raw = coco_eval.eval["precision"]  # [T, R, K, A, M]
    recalls_raw = coco_eval.eval["recall"]  # [T, K, A, M]
    precision_50_arr = precisions_raw[0, :, 0, 0, -1]
    recall_50_val = float(recalls_raw[0, 0, 0, -1])
    n = max(1, int(recall_50_val * 100))
    pre_mean = float(np.mean(precision_50_arr[:n]))
    f1 = 2 * pre_mean * recall_50_val / max(pre_mean + recall_50_val, 1e-12)
    return pre_mean, recall_50_val, f1


def save_coco_pr_curve(save_root: Path, coco_eval) -> Path:
    """导出 pycocotools 官方插值 PR 曲线（101 点），与 TDCNet 同格式。."""
    iou_thresholds = [round(float(x), 2) for x in np.arange(0.50, 0.96, 0.05)]
    recall_grid = np.linspace(0.0, 1.0, 101)
    prec = np.asarray(coco_eval.eval["precision"])  # [T, R, K, A, M]
    rec = np.asarray(coco_eval.eval["recall"])  # [T, K, A, M]

    curves = {}
    for t, thr in enumerate(iou_thresholds):
        p = prec[t, :, :, 0, 2]  # area=all, maxDets=100
        p_masked = np.where(p >= 0, p, np.nan)
        row = np.nanmean(p_masked, axis=1)
        row = np.where(np.isnan(row), -1.0, row)

        per_cat_recall = rec[t, :, 0, 2]
        valid_recall = per_cat_recall[per_cat_recall >= 0]
        max_recall = float(np.mean(valid_recall)) if valid_recall.size else -1.0

        curves[f"{thr:.2f}"] = {
            "recall": recall_grid.tolist(),
            "precision": row.tolist(),
            "max_recall": max_recall,
        }

    payload = {
        "schema_version": 2,
        "source": "pycocotools COCOeval",
        "interpolated": True,
        "recall_points": 101,
        "area": "all",
        "max_dets": 100,
        "curves": curves,
    }
    path = save_root / "pr_curve.json"
    save_json(path, payload)
    return path


def collect_per_image_errors(predictions, anns_by_image, image_ids, iou_threshold=0.5, conf_threshold=0.5):
    """按图统计漏检/误检（按 score 降序贪心匹配）。返回 {image_id: (n_miss, n_fp)}。."""
    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[int(p["image_id"])].append(p)

    errors = {}
    for img_id in image_ids:
        gts = anns_by_image.get(int(img_id), [])
        preds = sorted(pred_by_image.get(int(img_id), []), key=lambda x: -float(x.get("score", 0.0)))
        preds = [p for p in preds if float(p.get("score", 0.0)) >= conf_threshold]
        matched = [False] * len(gts)
        n_fp = 0
        for p in preds:
            pb = xywh_to_xyxy(p["bbox"])
            if not gts:
                n_fp += 1
                continue
            ious = np.array([bbox_iou_xyxy(pb, xywh_to_xyxy(a["bbox"])) for a in gts], dtype=np.float32)
            best_idx = None
            for gi in np.argsort(-ious):
                if ious[gi] >= iou_threshold and not matched[gi]:
                    best_idx = int(gi)
                    break
            if best_idx is None:
                n_fp += 1
            else:
                matched[best_idx] = True
        n_miss = sum(1 for m in matched if not m)
        if n_miss > 0 or n_fp > 0:
            errors[int(img_id)] = (n_miss, n_fp)
    return errors


def export_error_check_xlsx(save_root: Path, image_meta, per_image_errors) -> list:
    """按序列导出 error_check/<seq>.xlsx，含「漏检」「误检」两个 sheet（同 TDCNet）。."""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    seq_images = defaultdict(list)
    for im in image_meta:
        name = str(im.get("file_name", ""))
        seq_images[extract_sequence_name(name)].append({"id": int(im["id"]), "name": name})

    out_dir = save_root / "error_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    header_font = Font(bold=True)

    for seq in sorted(seq_images):
        rows = sorted(seq_images[seq], key=lambda r: r["id"])
        wb = Workbook()
        ws_miss = wb.active
        ws_miss.title = "漏检"
        ws_miss.append(["图像名", "image_id", "漏检目标数"])
        ws_fp = wb.create_sheet("误检")
        ws_fp.append(["图像名", "image_id", "误检框数"])

        for r in rows:
            n_miss, n_fp = per_image_errors.get(r["id"], (0, 0))
            base_name = os.path.basename(r["name"].replace("\\", "/"))
            if n_miss > 0:
                ws_miss.append([base_name, r["id"], n_miss])
            if n_fp > 0:
                ws_fp.append([base_name, r["id"], n_fp])

        for ws in (ws_miss, ws_fp):
            ws.column_dimensions["A"].width = 22
            ws.column_dimensions["B"].width = 12
            ws.column_dimensions["C"].width = 14
            for cell in ws[1]:
                cell.font = header_font

        safe_seq = re.sub(r'[\\/:*?"<>|]', "_", seq) or "default"
        path = out_dir / f"{safe_seq}.xlsx"
        wb.save(str(path))
        written.append(path)
    return written


def export_yolo_gt_as_coco(save_root: Path, image_records, anns_by_image, cat_ids, class_names) -> Path:
    """把 YOLO 标签形式的 GT 反写成一个 COCO JSON。.

    这样 --gt-format yolo 的数据集（如 MIST_det）也能直接喂 51_TDC/tools/eval_diag.py（它要求 --coco + --dets 都是 COCO 结构）。
    """
    images, annotations = [], []
    ann_id = 1
    for rec in image_records:
        images.append(
            {
                "id": int(rec["id"]),
                "file_name": str(rec["file_name"]),
                "width": int(rec["width"]),
                "height": int(rec["height"]),
            }
        )
        for ann in anns_by_image.get(int(rec["id"]), []):
            x, y, w, h = [float(v) for v in ann["bbox"]]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": int(rec["id"]),
                    "category_id": int(ann["category_id"]),
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    payload = {
        "info": {"description": "YOLO GT re-exported to COCO by eval_det.py"},
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": int(cid), "name": class_names[i] if i < len(class_names) else f"class_{cid}"}
            for i, cid in enumerate(cat_ids)
        ],
    }
    path = save_root / "gt_coco.json"
    save_json(path, payload)
    return path


# ===========================================================================
# GT Loading: COCO & YOLO
# ===========================================================================
def load_coco_gt(coco_json_path):
    """Return (image_records, anns_by_image, cat_ids, class_names).

    image_records: [{"id": int, "file_name": str, "width": int, "height": int}, ...] anns_by_image: {img_id: [{"bbox":
    [x,y,w,h], "category_id": int}, ...]}
    """
    coco = COCO(coco_json_path)
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    class_names = [coco.cats[c]["name"] for c in cat_ids] if cat_ids else ["target"]

    image_records = []
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        image_records.append(
            {
                "id": img_id,
                "file_name": info["file_name"],
                "width": info["width"],
                "height": info["height"],
            }
        )

    anns_by_image = defaultdict(list)
    for ann in coco.dataset.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        anns_by_image[int(ann["image_id"])].append(
            {
                "bbox": [float(v) for v in bbox],
                "category_id": ann["category_id"],
            }
        )

    return image_records, dict(anns_by_image), cat_ids, class_names


def load_yolo_gt(dataset_yaml_path, split="test"):
    """Load YOLO-format GT, returning the same structure as load_coco_gt."""
    # 用 PyYAML 直接解析：ultralytics 各版本对 `ultralytics.utils.YAML` 的导出不一致，
    # 不依赖它可避免 "cannot import name 'YAML'" 这类版本耦合问题。
    import yaml as _yaml

    with open(dataset_yaml_path, encoding="utf-8") as f:
        cfg = _yaml.safe_load(f)
    # cfg["path"] may be platform-specific (e.g. Windows path on Linux) —
    # fall back to the YAML file's directory when it doesn't exist
    data_root = Path(cfg["path"])
    if not data_root.is_dir():
        data_root = Path(dataset_yaml_path).resolve().parent
    # YAML values like "test/images" already include the images subdirectory
    img_subpath = cfg.get(split, "test")  # e.g. "test/images"
    img_dir = data_root / img_subpath
    lbl_dir = data_root / img_subpath.replace("images", "labels")
    names = cfg.get("names", {0: "target"})
    class_names = list(names.values())
    cat_ids = list(names.keys()) if names else [0]

    image_records = []
    anns_by_image = {}

    img_files = sorted(img_dir.glob("*"))
    # Filter to common image extensions
    img_files = [f for f in img_files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]

    for img_id, img_path in enumerate(img_files, start=1):
        # Get image dimensions
        with Image.open(img_path) as im:
            w, h = im.size
        # 只存文件名（YOLO 输出是扁平命名，如 62_0.bmp）——extract_sequence_name
        # 会据此回推出序列号 '62'；推理用绝对路径 _img_path，不依赖这个字段。
        rel_path = img_path.name

        image_records.append(
            {
                "id": img_id,
                "file_name": rel_path,
                "width": w,
                "height": h,
                "_img_path": str(img_path),  # absolute path for inference convenience
            }
        )

        # Load YOLO label
        label_path = lbl_dir / (img_path.stem + ".txt")
        anns = []
        if label_path.exists():
            with label_path.open("r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                    # Convert normalized → pixel coordinates (xywh)
                    anns.append(
                        {
                            "bbox": [
                                (cx - bw / 2) * w,
                                (cy - bh / 2) * h,
                                bw * w,
                                bh * h,
                            ],
                            "category_id": cat_ids[cls_id] if cls_id < len(cat_ids) else cat_ids[0],
                        }
                    )
        anns_by_image[img_id] = anns

    return image_records, anns_by_image, cat_ids, class_names


# ===========================================================================
# Evaluation metrics (unchanged)
# ===========================================================================
def evaluate_at_iou(predictions, gt_by_image, image_ids, iou_threshold):
    gt_boxes = {img_id: [xywh_to_xyxy(ann["bbox"]) for ann in gt_by_image.get(img_id, [])] for img_id in image_ids}
    matched = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in gt_boxes.items()}
    total_gt = int(sum(len(boxes) for boxes in gt_boxes.values()))

    sorted_preds = sorted(predictions, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    tp = np.zeros(len(sorted_preds), dtype=np.float32)
    fp = np.zeros(len(sorted_preds), dtype=np.float32)
    matched_ious = []

    for idx, pred in enumerate(sorted_preds):
        img_id = int(pred["image_id"])
        pred_box = xywh_to_xyxy(pred["bbox"])
        boxes = gt_boxes.get(img_id, [])
        if not boxes:
            fp[idx] = 1.0
            continue
        ious = np.array([bbox_iou_xyxy(pred_box, gt_box) for gt_box in boxes], dtype=np.float32)
        order = np.argsort(-ious)
        best_idx = None
        for gt_idx in order:
            if ious[gt_idx] >= iou_threshold and not matched[img_id][gt_idx]:
                best_idx = int(gt_idx)
                break
        if best_idx is None:
            fp[idx] = 1.0
        else:
            tp[idx] = 1.0
            matched[img_id][best_idx] = True
            matched_ious.append(float(ious[best_idx]))

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    f1_values = 2.0 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)
    if f1_values.size:
        best_idx = int(np.argmax(f1_values))
        best_precision = float(precisions[best_idx])
        best_recall = float(recalls[best_idx])
        best_f1 = float(f1_values[best_idx])
        best_score = float(sorted_preds[best_idx].get("score", 0.0))
        best_tp = int(tp_cum[best_idx])
        best_fp = int(fp_cum[best_idx])
        best_fn = max(0, total_gt - best_tp)
    else:
        best_precision = best_recall = best_f1 = best_score = 0.0
        best_tp = best_fp = 0
        best_fn = total_gt

    return {
        "ap": compute_ap(recalls, precisions),
        "iou_threshold": iou_threshold,
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
        "score": best_score,
        "tp": best_tp,
        "fp": best_fp,
        "fn": best_fn,
        "final_tp": int(tp.sum()),
        "final_fp": int(fp.sum()),
        "final_fn": max(0, total_gt - int(tp.sum())),
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "total_gt": total_gt,
        "num_predictions": len(sorted_preds),
        "recall_curve": recalls.astype(float).tolist(),
        "precision_curve": precisions.astype(float).tolist(),
    }


def evaluate_predictions(predictions, anns_by_image, image_ids, num_images):
    per_threshold = {}
    for thresh in COCO_IOU_THRESHOLDS:
        per_threshold[f"{thresh:.2f}"] = evaluate_at_iou(predictions, anns_by_image, image_ids, thresh)
    ap_values = [v["ap"] for v in per_threshold.values()]
    m50, m75 = per_threshold["0.50"], per_threshold["0.75"]
    return {
        "AP": float(np.mean(ap_values)) if ap_values else 0.0,
        "AP50": float(m50["ap"]),
        "AP75": float(m75["ap"]),
        "precision50": float(m50["precision"]),
        "recall50": float(m50["recall"]),
        "f1_50": float(m50["f1"]),
        "score50": float(m50["score"]),
        "mean_iou50": float(m50["mean_iou"]),
        "tp50": int(m50["tp"]),
        "fp50": int(m50["fp"]),
        "fn50": int(m50["fn"]),
        "final_tp50": int(m50["final_tp"]),
        "final_fp50": int(m50["final_fp"]),
        "final_fn50": int(m50["final_fn"]),
        "Pd": float(m50["recall"]),
        "Fa": float(m50["fp"]) / max(num_images, 1),
        "gt": int(m50["total_gt"]),
        "predictions": int(m50["num_predictions"]),
        "num_images": num_images,
        "per_iou_threshold": per_threshold,
    }


def build_pr_curve_payload(predictions, anns_by_image, image_ids):
    curves = {}
    for thresh in COCO_IOU_THRESHOLDS:
        result = evaluate_at_iou(predictions, anns_by_image, image_ids, thresh)
        curves[f"{thresh:.2f}"] = {
            "iou_threshold": thresh,
            "ap": result["ap"],
            "gt": result["total_gt"],
            "predictions": result["num_predictions"],
            "tp": result["final_tp"],
            "fp": result["final_fp"],
            "fn": result["final_fn"],
            "recall": result["recall_curve"],
            "precision": result["precision_curve"],
        }
    return {"schema_version": 1, "image_count": len(image_ids), "curves": curves}


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Unified detection evaluation (COCO or YOLO GT)")
    parser.add_argument(
        "--model",
        type=str,
        default="E:/Work/CV/Example/56_YOLO/runs/detect/runs/irdst/yolo26n_irdst/weights/best.pt",
        help="Path to trained YOLO weights",
    )
    # GT format
    parser.add_argument(
        "--gt-format",
        type=str,
        default="coco",
        choices=["coco", "yolo"],
        help="GT format: coco (COCO JSON) or yolo (YOLO label files)",
    )
    # COCO params
    parser.add_argument(
        "--coco-gt",
        type=str,
        default="E:/Work/Dataset/infrared_sequence/IRDST/IRDST_instances_test2017.json",
        help="Path to COCO GT JSON (--gt-format coco)",
    )
    parser.add_argument(
        "--dataset-img-path",
        type=str,
        default="E:/Work/Dataset/infrared_sequence/IRDST/images",
        help="Root dir of dataset images (COCO mode)",
    )
    # YOLO params
    parser.add_argument(
        "--data",
        type=str,
        default="E:/Work/Dataset/infrared_sequence/YOLO_VER/DAUB_YOLO/dataset.yaml",
        help="Path to YOLO dataset YAML (--gt-format yolo)",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split for YOLO GT (train/val/test)")
    # Output
    parser.add_argument("--save-path", type=str, default="runs/detect/eval", help="Directory for all output files")
    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "vis", "eval"],
        help="all=inference+evaluation, vis=inference only, eval=evaluation only",
    )
    parser.add_argument(
        "--eval-scope",
        type=str,
        default="gt",
        choices=["gt", "pred"],
        help="gt=评估完整 GT 图像列表（含背景帧，与 TDCNet 一致，默认）；pred=只评估有预测的图（旧口径）",
    )
    # Inference
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--device", type=str, default="0")
    # Error check
    parser.add_argument("--pr_conf", type=float, default=0.5, help="error_check 中判定漏检/误检使用的固定置信度阈值")
    # Vis
    parser.add_argument("--vis-conf", type=float, default=0.3)
    parser.add_argument("--max-vis-images", type=int, default=0)
    parser.add_argument(
        "--error-check",
        dest="error_check",
        action="store_true",
        default=True,
        help="导出 error_check/<seq>.xlsx（默认开启）",
    )
    parser.add_argument("--no-error-check", dest="error_check", action="store_false", help="不导出 error_check")
    parser.add_argument("--no-vis", action="store_true")
    return parser.parse_args()


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = parse_args()

    save_path = Path(args.save_path).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    do_inference = args.mode in ("all", "vis")
    do_eval = args.mode in ("all", "eval")
    do_vis = do_inference and not args.no_vis

    # --- Load GT ---
    if args.gt_format == "coco":
        if not _HAS_PYCOCOTOOLS:
            raise ImportError("COCO eval requires pycocotools. Run: pip install pycocotools")
        image_records, anns_by_image, cat_ids, class_names = load_coco_gt(args.coco_gt)
        dataset_name = Path(args.coco_gt).stem
        coco_gt_path = Path(args.coco_gt)
    else:
        image_records, anns_by_image, cat_ids, class_names = load_yolo_gt(args.data, args.split)
        dataset_name = Path(args.data).parent.name
        # YOLO GT 反写成 COCO，保证 eval_diag.py 之类的 COCO 工具也能用
        gt_coco_path = export_yolo_gt_as_coco(save_path, image_records, anns_by_image, cat_ids, class_names)
        coco_gt_path = gt_coco_path
        print(f"[INFO] Exported YOLO GT as COCO -> {gt_coco_path}")

    total_images = len(image_records)
    print(f"[INFO] GT format={args.gt_format}, {total_images} images, {len(cat_ids)} categories: {class_names}")

    # ------------------------------------------------------------------
    # Stage 1: Inference
    # ------------------------------------------------------------------
    eval_json_path = save_path / "eval_results.json"

    if do_inference:
        print("=" * 70)
        print("[STAGE 1] Running YOLO inference...")
        print("=" * 70)

        model = YOLO(args.model)

        vis_root = save_path / "vis"
        if do_vis:
            vis_root.mkdir(parents=True, exist_ok=True)
            font = _resolve_font()
            num_classes = max(len(cat_ids), 1)
            hsv_tuples = [(x / num_classes, 1.0, 1.0) for x in range(num_classes)]
            colors = [
                (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in (colorsys.hsv_to_rgb(*t) for t in hsv_tuples)
            ]

        predictions: list[dict] = []
        seq_counters: dict[str, int] = defaultdict(int)

        for rec in tqdm(image_records, desc="Inference", unit="img"):
            img_id = rec["id"]

            # Resolve image path
            if "_img_path" in rec:
                img_path = rec["_img_path"]
            else:
                img_path = os.path.join(args.dataset_img_path, rec["file_name"])

            results = model.predict(
                source=img_path, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False
            )

            if results[0].boxes is not None:
                boxes = results[0].boxes
                all_xyxy = boxes.xyxy.cpu().numpy()
                all_conf = boxes.conf.cpu().numpy()
                all_cls = boxes.cls.cpu().numpy().astype(int)
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = [float(v) for v in all_xyxy[i]]
                    predictions.append(
                        {
                            "image_id": img_id,
                            "category_id": cat_ids[all_cls[i]] if all_cls[i] < len(cat_ids) else cat_ids[0],
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(all_conf[i]),
                        }
                    )
            else:
                all_xyxy = np.empty((0, 4))
                all_conf = np.empty(0)
                all_cls = np.empty(0, dtype=int)

            # Visualization
            seq_name = extract_sequence_name(rec["file_name"])
            if do_vis:
                seq_counters[seq_name] += 1
                limit = args.max_vis_images
                if limit == 0 or seq_counters[seq_name] <= limit:
                    vis_mask = all_conf >= args.vis_conf
                    image = Image.open(img_path).convert("RGB")
                    if vis_mask.any():
                        annotated = draw_boxes_on_image(
                            image, all_xyxy[vis_mask], all_conf[vis_mask], all_cls[vis_mask], class_names, colors, font
                        )
                    else:
                        annotated = image
                    seq_out_dir = vis_root / seq_name
                    seq_out_dir.mkdir(parents=True, exist_ok=True)
                    annotated.save(seq_out_dir / f"{Path(rec['file_name']).stem}.png")

        save_json(eval_json_path, predictions)
        print(f"[INFO] Saved {len(predictions)} predictions -> {eval_json_path}")
        if do_vis:
            n_seqs = sum(1 for s in vis_root.iterdir() if s.is_dir()) if vis_root.exists() else 0
            print(f"[INFO] Saved visualizations for {n_seqs} sequences -> {vis_root}")

    # ------------------------------------------------------------------
    # Stage 2: Evaluation
    # ------------------------------------------------------------------
    if do_eval:
        print("\n" + "=" * 70)
        print("[STAGE 2] Running evaluation...")
        print("=" * 70)

        if not eval_json_path.exists():
            raise FileNotFoundError(f"Predictions not found: {eval_json_path}")

        with eval_json_path.open("r") as f:
            predictions = json.load(f)

        # --- 评估范围 ---
        # gt  : 完整 GT 图像列表（含无标注背景帧），与 TDCNet vid_infer_eval.py 一致
        # pred: 只评估有预测的图（本脚本旧口径）
        if args.eval_scope == "gt":
            image_ids = sorted(int(r["id"]) for r in image_records)
        else:
            image_ids = sorted({int(p["image_id"]) for p in predictions})
        id_set = set(image_ids)
        predictions = [p for p in predictions if int(p["image_id"]) in id_set]
        num_images = len(image_ids)
        n_with_pred = len({int(p["image_id"]) for p in predictions})
        print(
            f"[INFO] Evaluating on {num_images} images (scope={args.eval_scope}, "
            f"total GT {total_images}, images with predictions {n_with_pred})"
        )

        # 预测为空时给一条看得懂的错误，别让 pycocotools 抛 IndexError
        if not predictions:
            raise SystemExit(
                f"[错误] {eval_json_path} 里 0 个预测框，任何指标都算不出来。\n"
                f"       常见原因：推理阶段的 key 与 COCO file_name 的目录层级对不上，\n"
                f"       于是每个样本都被当成「找不到 image_id」跳过（DAUB 曾踩过这个坑）。\n"
                f"       线索：{eval_json_path.parent / 'infer_meta.json'} 里的 "
                f"images_without_gt_id / n_boxes。"
            )

        # COCO eval：coco 模式直接用 --coco-gt，yolo 模式用反写出的 gt_coco.json，
        # 因此四个检测数据集都能拿到与 TDCNet 同源的 COCO 官方指标。
        coco_stats = None
        coco_eval_obj = None
        if coco_gt_path is not None and _HAS_PYCOCOTOOLS:
            print("\n--- COCO Evaluation (pycocotools) ---")
            cocoGt = COCO(str(coco_gt_path))
            cocoDt = cocoGt.loadRes(str(eval_json_path))
            cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
            cocoEval.params.imgIds = image_ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            pre_coco, rec_coco, f1_coco = coco_pr_stats(cocoEval)
            coco_stats = {
                "AP": float(cocoEval.stats[0]),
                "AP50": float(cocoEval.stats[1]),
                "AP75": float(cocoEval.stats[2]),
                "AP_small": float(cocoEval.stats[3]),
                "AP_medium": float(cocoEval.stats[4]),
                "AP_large": float(cocoEval.stats[5]),
                "AR_1": float(cocoEval.stats[6]),
                "AR_10": float(cocoEval.stats[7]),
                "AR_100": float(cocoEval.stats[8]),
                "AR_small": float(cocoEval.stats[9]),
                "AR_medium": float(cocoEval.stats[10]),
                "AR_large": float(cocoEval.stats[11]),
                "precision50_coco": pre_coco,
                "recall50_coco": rec_coco,
                "f1_50_coco": f1_coco,
            }
            coco_eval_obj = cocoEval
            print(f"  AP_small: {coco_stats['AP_small']:.4f} | AR_100: {coco_stats['AR_100']:.4f}")
            print(f"  P/R/F1 (COCO PR curve @IoU0.50): {pre_coco:.4f} / {rec_coco:.4f} / {f1_coco:.4f}")

        # Custom eval (works for both COCO and YOLO GT)
        print("\n--- Custom Evaluation (Pd, Fa, mAP) ---")
        metrics = evaluate_predictions(predictions, anns_by_image, image_ids, num_images)

        print(f"  Images:           {num_images}")
        print(f"  GT boxes:         {metrics['gt']}")
        print(f"  Predictions:      {metrics['predictions']}")
        print(f"  AP (0.50:0.95):   {metrics['AP']:.4f}")
        print(f"  AP50:             {metrics['AP50']:.4f}")
        print(f"  AP75:             {metrics['AP75']:.4f}")
        print(f"  Precision@0.50:   {metrics['precision50']:.4f}")
        print(f"  Recall@0.50:      {metrics['recall50']:.4f}")
        print(f"  F1@0.50:          {metrics['f1_50']:.4f}")
        print(f"  Pd (detection):   {metrics['Pd']:.4f}")
        print(f"  Fa (FP/image):    {metrics['Fa']:.4f}")
        print(f"  TP/FP/FN@0.50:    {metrics['tp50']} / {metrics['fp50']} / {metrics['fn50']}")

        # ---- metrics.json：COCO 统计置于顶层（键名与 TDCNet 一致），自定义口径放进 metrics ----
        metrics_output = {
            "dataset": dataset_name,
            "model_path": args.model,
            "eval_scope": args.eval_scope,
            "images": num_images,
            "images_total": total_images,
            "images_skipped": total_images - num_images,
            "gt_format": args.gt_format,
            "confidence": args.conf,
            "nms_iou": args.iou,
            "pr_conf": args.pr_conf,
        }
        if coco_stats:
            metrics_output.update(coco_stats)
        metrics_output["metrics"] = metrics
        save_json(save_path / "metrics.json", metrics_output)
        print(f"\n[INFO] Saved metrics -> {save_path / 'metrics.json'}")

        # ---- PR 曲线：有官方 COCOeval 就用官方 101 点插值曲线（与 TDCNet 同源）----
        if coco_eval_obj is not None:
            pr_path = save_coco_pr_curve(save_path, coco_eval_obj)
            print(f"[INFO] Saved official COCO PR curve -> {pr_path}")
            # 自定义贪心口径的曲线留档，便于对照
            pr_custom = build_pr_curve_payload(predictions, anns_by_image, image_ids)
            save_json(save_path / "pr_curve_custom.json", pr_custom)
        else:
            pr_payload = build_pr_curve_payload(predictions, anns_by_image, image_ids)
            save_json(save_path / "pr_curve.json", pr_payload)
            print(f"[INFO] Saved PR curve data -> {save_path / 'pr_curve.json'}")

        # ---- summary.csv：TDCNet 口径（只含 COCO 统计）----
        if coco_stats:
            summary_path = save_path / "summary.csv"
            coco_row = {"dataset": dataset_name, "images": num_images, **coco_stats}
            with summary_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COCO_SUMMARY_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(coco_row)
            print(f"[INFO] Saved summary (TDCNet 口径) -> {summary_path}")

        # ---- summary_custom.csv：Pd/Fa/自定义 best-F1 口径 ----
        custom_path = save_path / "summary_custom.csv"
        custom_row = {
            "dataset": dataset_name,
            "images": num_images,
            "gt": metrics["gt"],
            "predictions": metrics["predictions"],
            "AP": metrics["AP"],
            "AP50": metrics["AP50"],
            "AP75": metrics["AP75"],
            "precision50": metrics["precision50"],
            "recall50": metrics["recall50"],
            "f1_50": metrics["f1_50"],
            "score50": metrics["score50"],
            "Pd": metrics["Pd"],
            "Fa": metrics["Fa"],
            "mean_iou50": metrics["mean_iou50"],
            "tp50": metrics["tp50"],
            "fp50": metrics["fp50"],
            "fn50": metrics["fn50"],
        }
        try:
            with custom_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
                writer.writeheader()
                writer.writerow(custom_row)
            print(f"[INFO] Saved summary (Pd/Fa 口径) -> {custom_path}")
        except PermissionError:
            alt = custom_path.with_name(f"summary_custom_{dataset_name}.csv")
            with alt.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
                writer.writeheader()
                writer.writerow(custom_row)
            print(f"[WARN] summary_custom.csv locked. Saved to -> {alt}")

        # ---- error_check/<seq>.xlsx：逐序列漏检/误检（与 TDCNet 一致）----
        if args.error_check:
            print("\n--- Export per-sequence error check (xlsx) ---")
            per_image_errors = collect_per_image_errors(
                predictions,
                anns_by_image,
                image_ids,
                iou_threshold=0.5,
                conf_threshold=args.pr_conf,
            )
            eval_id_set = set(image_ids)
            eval_meta = [
                {"id": int(r["id"]), "file_name": r["file_name"]} for r in image_records if int(r["id"]) in eval_id_set
            ]
            try:
                xlsx_files = export_error_check_xlsx(save_path, eval_meta, per_image_errors)
                n_img_miss = sum(1 for v in per_image_errors.values() if v[0] > 0)
                n_img_fp = sum(1 for v in per_image_errors.values() if v[1] > 0)
                print(
                    f"[INFO] images with miss: {n_img_miss}, images with FP: {n_img_fp}; "
                    f"exported {len(xlsx_files)} xlsx -> {save_path / 'error_check'}"
                )
            except ImportError:
                print("[WARN] openpyxl 未安装，跳过 error_check 导出（pip install openpyxl）")

    print("\n" + "=" * 70)
    print("[DONE]")
    print(f"  Output directory: {save_path}")
    if do_eval:
        print(f"  Metrics JSON:     {save_path / 'metrics.json'}")
        print(f"  PR curve JSON:    {save_path / 'pr_curve.json'}")
        print(f"  Summary CSV:      {save_path / 'summary.csv'}")
        print(f"  Summary (custom): {save_path / 'summary_custom.csv'}")
        if args.error_check:
            print(f"  Error check:      {save_path / 'error_check'}")
        if args.gt_format == "yolo":
            print(f"  GT (COCO copy):   {save_path / 'gt_coco.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
