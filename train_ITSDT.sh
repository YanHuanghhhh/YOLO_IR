#!/bin/bash
# Train yolo26n on IRDST infrared dataset
#
# Usage:
#   bash train_irdst.sh                          # default params
#   bash train_irdst.sh --epochs 200 --batch 32  # override params
#   bash train_irdst.sh --device 1               # different GPU

cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
python train_irdst.py \
    --data E:/Work/Dataset/infrared_sequence/YOLO_VER/ITSDT_YOLO/dataset.yaml \
    --model ../weights/yolo26n.pt \
    --epochs 200 \
    --imgsz 640 \
    --batch 32 \
    --device 0 \
    "$@"
