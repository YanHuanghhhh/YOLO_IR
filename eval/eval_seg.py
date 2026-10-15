# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Segmentation evaluation for NUDT-MIRSDT-v2 / infrared datasets.

Mirrors FeedbackSTS-Det evaluate.py metrics: mIoU, Pd, Fa, AUC, ROC curve.

Stages:
  1. YOLO inference on test images → raw prediction maps (.npy)
  2. Save binary visualization masks (single-channel png, 0/255)
  3. Evaluate vs GT masks → metrics.json, roc.mat, summary.csv

Usage:
    python eval/eval_seg.py                                    # inference + vis + eval
    python eval/eval_seg.py --mode vis                         # inference + vis only
    python eval/eval_seg.py --mode eval --no-vis               # eval only (skip inference)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.integrate import trapezoid
from tqdm import tqdm

from ultralytics import YOLO

try:
    from scipy.io import savemat as _savemat
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ===========================================================================
# Metric classes (ported from FeedbackSTS-Det evaluation/)
# ===========================================================================
class _ROCMetric:
    """ROC curve metric: false/true detection rates across thresholds."""

    def __init__(self, bins: int = 200):
        self.bins = bins
        self.reset()

    def update(self, pred: np.ndarray, label: np.ndarray) -> None:
        if pred.max() > 0:
            pred = pred / pred.max()
        label = label.astype(np.uint8)

        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(label)
        if num_labels <= 1:
            return

        back_mask = labels == 0
        self.background_area += int(np.sum(back_mask))
        self.target_nums += num_labels - 1

        for ibin in range(self.bins + 1):
            thre = ibin / self.bins
            pred_binary = pred >= thre
            self.false_detect[ibin] += int(np.sum(np.logical_and(back_mask, pred_binary)))
            for t in range(1, num_labels):
                target_mask = labels == t
                self.true_detect[ibin] += np.sum(np.logical_and(target_mask, pred_binary)) > 0

    def get(self):
        fpr = self.false_detect / max(self.background_area, 1)
        tpr = self.true_detect / max(self.target_nums, 1)
        auc = float(trapezoid(x=np.flip(np.append(fpr, 0)), y=np.flip(np.append(tpr, 0))))
        return fpr, tpr, auc

    def reset(self):
        self.false_detect = np.zeros(self.bins + 1, dtype=np.float64)
        self.true_detect = np.zeros(self.bins + 1, dtype=np.float64)
        self.background_area = 0
        self.target_nums = 0


class _mIoUMetric:
    """Pixel accuracy + mIoU (single-class foreground)."""

    def __init__(self):
        self.reset()

    def update(self, preds: np.ndarray, labels: np.ndarray) -> None:
        pred = (preds > 0).astype(np.float64)
        label = labels.astype(np.float64)
        self.total_correct += int(np.sum((pred == label) * (label > 0)))
        self.total_label += int(np.sum(label > 0))
        inter = int(np.sum(pred * (pred == label)))
        union = int(np.sum(pred) + np.sum(label) - inter)
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pix_acc = 1.0 * self.total_correct / max(self.total_label, 1)
        iou = 1.0 * self.total_inter / max(self.total_union, 1)
        return float(pix_acc), float(iou)

    def reset(self):
        self.total_correct = 0
        self.total_label = 0
        self.total_inter = 0
        self.total_union = 0


class _PDFAMetric:
    """Detection-level Pd/Fa via connected-component matching (the 'our' version)."""

    def __init__(self):
        self.reset()

    def update(self, pred: np.ndarray, label: np.ndarray) -> None:
        if pred.max() > 0:
            pred = pred / pred.max()
        label = label.astype(np.uint8)

        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(label)
        if num_labels <= 1:
            return

        back_mask = labels == 0
        self.background_area += int(np.sum(back_mask))
        self.target_nums += num_labels - 1

        pred_binary = pred > 0.5
        self.false_detect += int(np.sum(np.logical_and(back_mask, pred_binary)))
        for t in range(1, num_labels):
            target_mask = labels == t
            self.true_detect += np.sum(np.logical_and(target_mask, pred_binary)) > 0

    def get(self):
        fa = self.false_detect / max(self.background_area, 1)
        pd = self.true_detect / max(self.target_nums, 1)
        return float(pd), float(fa)

    def reset(self):
        self.false_detect = 0
        self.true_detect = 0
        self.background_area = 0
        self.target_nums = 0


class _TPFNFP:
    """TP/FP/FN for computing mIoU, precision, recall, F1."""

    def __init__(self):
        self.reset()

    def update(self, labels: np.ndarray, preds: np.ndarray) -> None:
        if preds.max() > 0:
            preds = preds / preds.max() > 0.5
        preds = preds.astype(np.int64)
        if labels.max() > 0:
            labels = (labels / labels.max()).astype(np.int64)
        else:
            labels = labels.astype(np.int64)
        tp = int(np.sum(preds * (preds == labels)))
        fp = int(np.sum(preds)) - tp
        fn = int(np.sum(labels)) - tp
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn

    def get(self):
        tp, fp, fn = self.total_tp, self.total_fp, self.total_fn
        denom = tp + fp + fn
        miou = 1.0 * tp / max(denom, 1)
        prec = 1.0 * tp / max(tp + fp, 1)
        recall = 1.0 * tp / max(tp + fn, 1)
        fscore = 2.0 * prec * recall / max(prec + recall, 1e-12)
        return float(miou), float(prec), float(recall), float(fscore)

    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0


# ===========================================================================
# Helpers
# ===========================================================================
def _normalize(img: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1]."""
    max_val = img.max()
    return img.astype(np.float64) / max_val if max_val > 0 else img.astype(np.float64)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation evaluation (NUDT-MIRSDT-v2, etc.)")
    # Model
    parser.add_argument("--model", type=str,
                        default="E:/Work/CV/Example/56_YOLO/runs/seg/yolo26n-sem_nudt/weights/best.pt",
                        help="Path to trained YOLO segmentation weights")
    # Dataset
    parser.add_argument("--dataset-dir", type=str,
                        default="E:/Work/Dataset/infrared_sequence/NUDT-MIRSDT-NEW-v2",
                        help="Root directory of dataset (contains images/, masks/, video_idx/)")
    parser.add_argument("--img-dir", type=str, default="images",
                        help="Subdirectory for images (default: images)")
    parser.add_argument("--mask-dir", type=str, default="masks",
                        help="Subdirectory for GT masks (NUDT: masks, MIST: mask)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split name (matches video_idx/<split>.txt or root <split>.txt)")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Explicit path to split file (overrides --split)")
    # Output
    parser.add_argument("--save-path", type=str, default="runs/seg/eval",
                        help="Directory for all output files")
    # Mode
    parser.add_argument("--mode", type=str, default="all", choices=["all", "vis", "eval"],
                        help="all=inference+vis+mat+eval, vis=inference+vis+mat, eval=eval only")
    # Inference params
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold for mask prediction")
    parser.add_argument("--device", type=str, default="0", help="GPU device")
    # Threshold
    parser.add_argument("--thre", type=float, default=0.5,
                        help="Binarization threshold for evaluation")
    # Visualization / .mat toggles
    parser.add_argument("--no-vis", action="store_true", help="Skip saving visualization masks")
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

    # ------------------------------------------------------------------
    # Load sequence list
    # ------------------------------------------------------------------
    if args.split_file:
        txt_path = Path(args.split_file)
    else:
        txt_path = Path(args.dataset_dir) / "video_idx" / f"{args.split}.txt"
        if not txt_path.exists():
            # Fallback: try root directory
            txt_path = Path(args.dataset_dir) / f"{args.split}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Split file not found: {txt_path}")
    with txt_path.open("r") as f:
        seq_list = [line.strip() for line in f.read().splitlines() if line.strip()]
    print(f"[INFO] {len(seq_list)} sequences in {args.split} split")

    # ------------------------------------------------------------------
    # Stage 1: Inference + save vis/mat
    # ------------------------------------------------------------------
    preds_dir = save_path / "preds"
    vis_dir = save_path / "vis"

    if do_inference:
        print("=" * 70)
        print("[STAGE 1] Running YOLO segmentation inference...")
        print("=" * 70)

        model = YOLO(args.model)
        preds_dir.mkdir(parents=True, exist_ok=True)
        if do_vis:
            vis_dir.mkdir(parents=True, exist_ok=True)

        for seq in tqdm(seq_list, desc="Inference", unit="seq"):
            img_seq_dir = Path(args.dataset_dir) / args.img_dir / seq
            mask_seq_dir = Path(args.dataset_dir) / args.mask_dir / seq

            if not img_seq_dir.exists():
                print(f"[WARN] Image dir not found: {img_seq_dir}, skipping")
                continue

            # Prepare output dirs
            (preds_dir / seq).mkdir(parents=True, exist_ok=True)
            if do_vis:
                (vis_dir / seq).mkdir(parents=True, exist_ok=True)

            img_files = sorted(img_seq_dir.iterdir(), key=lambda x: x.name)
            for img_path in img_files:
                name_stem = img_path.stem

                # Run YOLO prediction
                results = model.predict(
                    source=str(img_path),
                    imgsz=args.imgsz,
                    conf=args.conf,
                    device=args.device,
                    verbose=False,
                    retina_masks=True,
                )

                # Extract raw prediction map
                raw_map = _extract_pred_map(results)
                if raw_map is None:
                    continue

                # Save raw prediction (npy for fast reload during eval)
                np.save(str(preds_dir / seq / f"{name_stem}.npy"), raw_map)

                # Save binary visualization mask: single-channel, 0 or 255
                if do_vis:
                    pred_norm = _normalize(raw_map)
                    vis_mask = (pred_norm > args.thre).astype(np.uint8) * 255
                    Image.fromarray(vis_mask).save(vis_dir / seq / f"{name_stem}.png")

        print(f"[INFO] Saved raw predictions (.npy) -> {preds_dir}")
        if do_vis:
            print(f"[INFO] Saved visualization masks -> {vis_dir}")

    # ------------------------------------------------------------------
    # Stage 2: Evaluation
    # ------------------------------------------------------------------
    if do_eval:
        print("\n" + "=" * 70)
        print("[STAGE 2] Running evaluation...")
        print("=" * 70)

        if not preds_dir.exists():
            raise FileNotFoundError(f"Predictions not found: {preds_dir}\nRun with --mode all or --mode vis first.")

        roc = _ROCMetric(bins=200)
        eval_miou = _mIoUMetric()
        eval_pd_fa = _PDFAMetric()
        eval_tpfn = _TPFNFP()

        for seq in tqdm(seq_list, desc="Evaluating", unit="seq"):
            mask_seq_dir = Path(args.dataset_dir) / args.mask_dir / seq
            pred_seq_dir = preds_dir / seq

            if not pred_seq_dir.exists():
                continue

            pred_files = sorted(pred_seq_dir.glob("*.npy"), key=lambda x: x.name)
            for pred_file in pred_files:
                name_stem = pred_file.stem
                mask_path = mask_seq_dir / f"{name_stem}.png"

                if not mask_path.exists():
                    continue

                # Load prediction and mask
                raw_map = np.load(str(pred_file))
                mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float64)

                # Resize if needed
                if raw_map.shape != mask.shape:
                    raw_map = cv2.resize(raw_map, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

                pred_norm = _normalize(raw_map)
                mask_norm = _normalize(mask)

                roc.update(pred_norm.copy(), mask.copy())
                eval_miou.update(pred_norm.copy(), mask_norm)
                eval_pd_fa.update(pred_norm.copy(), mask.copy())
                eval_tpfn.update(labels=mask.copy(), preds=pred_norm.copy())

        # Collect results
        pix_acc, miou_val = eval_miou.get()
        fpr, tpr, auc_val = roc.get()
        pd_val, fa_val = eval_pd_fa.get()
        miou_tp, prec, recall, fscore = eval_tpfn.get()

        print(f"\n{'='*50}")
        print(f"  pixAcc:        {pix_acc:.6f}")
        print(f"  mIoU:          {miou_val:.6f}")
        print(f"  mIoU (TPFNFP): {miou_tp:.6f}")
        print(f"  AUC:           {auc_val:.6f}")
        print(f"  Pd:            {pd_val:.6f}")
        print(f"  Fa:            {fa_val:.8f}")
        print(f"  Precision:     {prec:.6f}")
        print(f"  Recall:        {recall:.6f}")
        print(f"  F1:            {fscore:.6f}")
        print(f"{'='*50}")

        # --- Save metrics.json ---
        metrics = {
            "pixAcc": pix_acc,
            "mIoU": miou_val,
            "mIoU_tpfp": miou_tp,
            "AUC": auc_val,
            "Pd": pd_val,
            "Fa": fa_val,
            "Precision": prec,
            "Recall": recall,
            "F1": fscore,
            "threshold": args.thre,
        }
        _save_json(save_path / "metrics.json", metrics)
        print(f"\n[INFO] Saved metrics -> {save_path / 'metrics.json'}")

        # --- Save ROC .mat ---
        if _HAS_SCIPY:
            _savemat(
                str(save_path / "roc.mat"),
                {
                    "fpr": fpr.astype(np.float64),
                    "tpr": tpr.astype(np.float64),
                    "auc": auc_val,
                    "pd": pd_val,
                    "fa": fa_val,
                    "mIoU": miou_val,
                    "mIoU_tpfp": miou_tp,
                    "prec": prec,
                    "recall": recall,
                    "fscore": fscore,
                    "pixAcc": pix_acc,
                },
            )
            print(f"[INFO] Saved ROC data -> {save_path / 'roc.mat'}")

        # --- Save summary.csv ---
        summary_path = save_path / "summary.csv"
        row = {
            "dataset": Path(args.dataset_dir).name,
            "split": args.split,
            "pixAcc": pix_acc,
            "mIoU": miou_val,
            "mIoU_tpfp": miou_tp,
            "AUC": auc_val,
            "Pd": pd_val,
            "Fa": fa_val,
            "Precision": prec,
            "Recall": recall,
            "F1": fscore,
        }
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        print(f"[INFO] Saved summary -> {summary_path}")

    # --- Done ---
    print("\n" + "=" * 70)
    print("[DONE]")
    print(f"  Output directory: {save_path}")
    if do_inference:
        print(f"  Raw predictions:  {preds_dir}")
        if do_vis:
            print(f"  Vis masks:        {vis_dir}")
    if do_eval:
        print(f"  Metrics JSON:     {save_path / 'metrics.json'}")
        print(f"  ROC .mat:         {save_path / 'roc.mat'}")
        print(f"  Summary CSV:      {save_path / 'summary.csv'}")
    print("=" * 70)


# ===========================================================================
# Prediction map extraction
# ===========================================================================
def _extract_pred_map(results) -> np.ndarray | None:
    """Extract a single raw prediction map from YOLO segmentation results.

    Handles both semantic segmentation (semantic_mask, shape H×W) and instance
    segmentation (masks, shape N×H×W merged via max).
    """
    result = results[0]

    # 1. Semantic segmentation: output is in result.semantic_mask.data
    if hasattr(result, "semantic_mask") and result.semantic_mask is not None:
        data = result.semantic_mask.data
        if data is not None:
            return data.cpu().numpy().astype(np.float64)

    # 2. Instance segmentation: merge all instance masks
    if hasattr(result, "masks") and result.masks is not None:
        data = result.masks.data
        if data is not None and len(data) > 0:
            combined = data.cpu().numpy().max(axis=0)
            return combined.astype(np.float64)

    # 3. Classification probs (unlikely for segmentation)
    if hasattr(result, "probs") and result.probs is not None:
        data = result.probs.data
        if data is not None:
            return data.cpu().numpy().astype(np.float64)

    return None


if __name__ == "__main__":
    main()
