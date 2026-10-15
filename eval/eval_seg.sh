#!/bin/bash
# NUDT-MIRSDT-v2 segmentation evaluation — inference + vis + metrics (mIoU, Pd, Fa, AUC, ROC)
#
# Usage:
#   bash eval/eval_seg.sh                                   # inference + vis + eval (default)
#   bash eval/eval_seg.sh --mode vis                        # inference + vis only
#   bash eval/eval_seg.sh --mode eval --no-vis              # eval only (skip inference)
#   bash eval/eval_seg.sh --thre 0.7                        # custom binarization threshold

cd "$(dirname "$0")/.."
PYTHONPATH=. python eval/eval_seg.py \
  --model ../runs/segment/runs/seg/yolo26n_seg_nudt/weights/best.pt \
  --dataset-dir E:/Work/Dataset/infrared_sequence/NUDT-MIRSDT-NEW-v2 \
  --split "test" \
  --save-path ../runs/segment/runs/seg/yolo26n_seg_nudt/eval \
  --imgsz 640 \
  --conf 0.001 \
  --thre 0.5 \
  --device 0 \
  "$@"
