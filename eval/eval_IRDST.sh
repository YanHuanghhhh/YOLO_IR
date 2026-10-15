#!/bin/bash
# Detection eval on IRDST (COCO GT)
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE
PYTHONPATH=. python eval/eval_det.py \
  --gt-format coco \
  --model /project/IDIP/HYA/code/56_YOLO/YOLO_IR/runs/detect/runs/detect/yolo26n_irdst_det/weights/best.pt \
  --coco-gt /project/IDIP/Dataset/IR_SEQ/IRDST/IRDST_instances_test2017.json \
  --dataset-img-path /project/IDIP/Dataset/IR_SEQ/IRDST \
  --save-path /project/IDIP/HYA/code/56_YOLO/YOLO_IR/runs/detect/runs/detect/yolo26n_irdst_det/eval \
  --imgsz 640 \
  --conf 0.001 \
  --iou 0.55 \
  --device 3 \
  "$@"
