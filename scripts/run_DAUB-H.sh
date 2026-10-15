#!/bin/bash
# DAUB-H：各尺度「训练 → 评估」一键跑完（默认 s,m,l,x）
#
#   bash scripts/run_DAUB-H.sh                  # 全部尺度
#   bash scripts/run_DAUB-H.sh --scales s       # 只跑单个尺度
#   bash scripts/run_DAUB-H.sh --scales s,m     # 跑指定多个尺度
#   bash scripts/run_DAUB-H.sh --device 2
#
# 数据集：YOLO_VER/DAUB-H_YOLO（8596 train / 5181 test），
#         评估用原始 COCO GT：DAUB-H/annotations/instances_test2017.json
# 注意：test 里 500 帧是"背景帧"（无框，正常当负样本）；另有 504 帧完全不在
#       JSON 里（data16 293-498、data19 1301-1598），已确认含目标，默认不纳入。
#
# 评估默认不生成 vis；需要时加 --vis。更多参数见 scripts/run_scales.sh --help
cd "$(dirname "$0")/.." || exit 1
exec bash scripts/run_scales.sh --dataset DAUB-H "$@"
