#!/bin/bash
# Train segmentation on MIST
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE
python train_seg.py \
  --data /project/IDIP/Dataset/IR_SEQ/YOLO_VER/MIST_YOLO_seg/dataset.yaml \
  --model ../weights/yolo26s-seg.pt \
  --name yolo26n-seg_mist \
  --epochs 300 \
  --imgsz 640 \
  --batch 96 \
  --device 6 \
  "$@"
