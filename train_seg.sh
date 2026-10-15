#!/bin/bash
# Train YOLO semantic segmentation on NUDT-MIRSDT-v2
#
# Usage:
#   bash train_seg.sh                                    # default: yolo26n-sem
#   bash train_seg.sh --model ../weights/yolo26s-sem.pt  # small model (better accuracy)
#   bash train_seg.sh --epochs 200 --batch 32            # custom params

cd "$(dirname "$0")"
python train_seg.py \
  --data E:/Work/Dataset/infrared_sequence/YOLO_VER/NUDT_YOLO/dataset.yaml \
  --model ../weights/yolo26s-seg.pt \
  --name yolo26n_seg_nudt \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --device 0 \
  "$@"
