#!/bin/bash
# Segmentation eval on MIST
# MIST uses: image/ (not images), mask/ (not masks), val_all.txt at root (not video_idx/)
cd "$(dirname "$0")/.."
PYTHONPATH=. python eval/eval_seg.py \
  --model /project/IDIP/HYA/code/56_YOLO/YOLO_IR/runs/segment/runs/seg/yolo26n-seg_mist/weights/best.pt \
  --dataset-dir /project/IDIP/Dataset/IR_SEQ/MIST \
  --img-dir image \
  --mask-dir mask \
  --split val_all \
  --save-path /project/IDIP/HYA/code/56_YOLO/YOLO_IR/runs/segment/runs/seg/yolo26n-seg_mist/eval \
  --imgsz 640 \
  --conf 0.001 \
  --thre 0.5 \
  --device 6 \
  "$@"
