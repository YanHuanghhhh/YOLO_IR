#!/bin/bash
# DAUB：各尺度「训练 → 评估」一键跑完（默认 s,m,l,x）
#
#   bash scripts/run_DAUB.sh                   # 全部尺度
#   bash scripts/run_DAUB.sh --scales x        # 只跑单个尺度
#   bash scripts/run_DAUB.sh --scales s,m      # 跑指定多个尺度
#
# 评估默认不生成 vis；需要时加 --vis。更多参数见 scripts/run_scales.sh --help
cd "$(dirname "$0")/.." || exit 1
exec bash scripts/run_scales.sh --dataset DAUB "$@"
