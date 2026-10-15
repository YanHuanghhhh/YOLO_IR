#!/bin/bash
# Segmentation eval on NUDT-MIRSDT-v2
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE
PYTHONPATH=. python eval/eval_seg.py \
  --model /project/IDIP/HYA/code/56_YOLO/YOLO_IR/runs/segment/runs/seg/yolo26n-seg_nudt/weights/best.pt \
  --dataset-dir /project/IDIP/Dataset/IR_SEQ/NUDT-MIRSDT-NEW-v2 \
  --split test \
  --save-path /project/IDIP/HYA/code/56_YOLO/YOLO_IR/runs/segment/runs/seg/yolo26n-seg_nudt/eval \
  --imgsz 640 \
  --conf 0.001 \
  --thre 0.5
--device 7 \
  "$@"
