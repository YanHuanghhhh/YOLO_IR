#!/bin/bash
# IRDST-R：各尺度「训练 → 评估」一键跑完（默认 s,m,l,x）
#
#   bash scripts/run_IRDST-R.sh                 # 全部尺度
#   bash scripts/run_IRDST-R.sh --scales s      # 只跑单个尺度
#   bash scripts/run_IRDST-R.sh --device 3
#
# 数据集：YOLO_VER/IRDST-R_YOLO（11768 train / 7339 test），
#         评估用原始 COCO GT：IRDST-R/IRDST1_anns/instances_test2017.json
# 注意：train 的 seq61(182 张) 在原 JSON 里 width/height 写错成 720x480
#       （实际 934x696），转换脚本已按磁盘真实尺寸归一化；test 是干净的。
#       另外 images/ 下还有 55 个序列不属于任何 split。
#
# 评估默认不生成 vis；需要时加 --vis。更多参数见 scripts/run_scales.sh --help
cd "$(dirname "$0")/.." || exit 1
exec bash scripts/run_scales.sh --dataset IRDST-R "$@"
