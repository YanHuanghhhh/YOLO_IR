#!/bin/bash
# Train segmentation on NUDT-MIRSDT-v2
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE
python train_seg.py \
  --data /project/IDIP/Dataset/IR_SEQ/YOLO_VER/NUDT_YOLO/dataset.yaml \
  --model ../weights/yolo26s-seg.pt \
  --name yolo26n-seg_nudt \
  --epochs 300 \
  --imgsz 640 \
  --batch 32 \
  --device 7 \
  "$@"
