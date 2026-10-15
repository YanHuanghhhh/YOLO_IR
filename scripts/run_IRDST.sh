#!/bin/bash
# IRDST：各尺度「训练 → 评估」一键跑完（默认 s,m,l,x）
#
#   bash scripts/run_IRDST.sh                  # 全部尺度
#   bash scripts/run_IRDST.sh --scales l       # 只跑单个尺度
#   bash scripts/run_IRDST.sh --scales s,l     # 跑指定多个尺度
#   bash scripts/run_IRDST.sh --device 2 --epochs 300
#
# 评估默认不生成 vis；需要时加 --vis。更多参数见 scripts/run_scales.sh --help
cd "$(dirname "$0")/.." || exit 1
exec bash scripts/run_scales.sh --dataset IRDST "$@"
