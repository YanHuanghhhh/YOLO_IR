#!/bin/bash
# Detection eval on MIST (YOLO GT)
cd "$(dirname "$0")/.."
PYTHONPATH=. python eval/eval_det.py \
  --gt-format yolo \
  --model /project/IDIP/HYA/code/56_YOLO/YOLO_IR/runs/detect/runs/detect/yolo26n_mist_det/weights/best.pt \
  --data /project/IDIP/Dataset/IR_SEQ/YOLO_VER/MIST_YOLO_det/dataset.yaml \
  --split test \
  --save-path /project/IDIP/HYA/code/56_YOLO/YOLO_IR/runs/detect/runs/detect/yolo26n_mist_det/eval \
  --imgsz 640 \
  --conf 0.001 \
  --iou 0.55 \
  --device 5 \
  "$@"
