#!/bin/bash
# Train detection on MIST
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE
python train_det.py \
  --data /project/IDIP/Dataset/IR_SEQ/YOLO_VER/MIST_YOLO_det/dataset.yaml \
  --model ../weights/yolo26n.pt \
  --name yolo26n_mist_det \
  --epochs 300 \
  --imgsz 640 \
  --batch 24 \
  --device 5 \
  "$@"
