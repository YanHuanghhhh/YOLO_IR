#!/bin/bash
# IRDST YOLO inference
#
# Usage:
#   bash predict_irdst.sh                          # all test images (default)
#   bash predict_irdst.sh --source image.jpg        # single image
#   bash predict_irdst.sh --conf 0.5 --save-txt     # custom conf + save labels

cd "$(dirname "$0")"
python predict_irdst.py \
  --model E:/Work/CV/Example/56_YOLO/runs/detect/runs/irdst/yolo26n_irdst/weights/best.pt \
  --source E:/Work/Dataset/infrared_sequence/YOLO_VER/IRDST_YOLO/test/images \
  --imgsz 640 \
  --conf 0.25 \
  --device 0 \
  "$@"
