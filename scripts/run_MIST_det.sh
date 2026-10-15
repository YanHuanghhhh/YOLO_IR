#!/bin/bash
# MIST（检测）：各尺度「训练 → 评估」一键跑完（默认 s,m,l,x）
#
#   bash scripts/run_MIST_det.sh               # 全部尺度
#   bash scripts/run_MIST_det.sh --scales s    # 只跑单个尺度
#   bash scripts/run_MIST_det.sh --device 4
#
# 注意：MIST 的 GT 是 YOLO 标签格式，评估走 --gt-format yolo，
#       脚本会自动把 GT 反写成 gt_coco.json 再跑 COCO 官方指标。
# 评估默认不生成 vis；需要时加 --vis。更多参数见 scripts/run_scales.sh --help
cd "$(dirname "$0")/.." || exit 1
exec bash scripts/run_scales.sh --dataset MIST_det "$@"
