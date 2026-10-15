#!/bin/bash
# =============================================================================
#  单数据集 · 多尺度 YOLO11 「训练 → 评估」一体化驱动
#
#  一个脚本跑完某个数据集的所有尺度：每个尺度先训练（train_det.py），
#  训练结束后立刻自动评估（eval/eval_det.py），再进入下一个尺度。
#  评估默认 **不生成 vis 文件夹**（--no-vis），需要时加 --vis。
#
#  用法（在仓库根目录 YOLO_IR 下执行；本地与服务器同一套脚本）：
#
#    bash scripts/run_scales.sh --dataset IRDST                  # 默认 s,m,l,x 全跑
#    bash scripts/run_scales.sh --dataset IRDST --scales l       # 只跑单个尺度
#    bash scripts/run_scales.sh --dataset IRDST --scales s,l     # 跑指定多个尺度
#    bash scripts/run_scales.sh --dataset MIST_det --scales s --device 2
#    bash scripts/run_scales.sh --dataset DAUB-H --scales s --device 2
#    bash scripts/run_scales.sh --dataset IRDST-R --scales s --device 3
#    bash scripts/run_scales.sh --dataset IRDST --list           # 只看会执行什么
#    bash scripts/run_scales.sh --dataset IRDST --dry-run        # 只打印命令不执行
#  数据集：IRDST、DAUB、ITSDT-15K、MIST_det、DAUB-H、IRDST-R
#
#  本地冒烟测试（临时覆盖数据集根；服务器上用默认值即可）：
#    DATA_ROOT=E:/Work/Dataset/infrared_sequence \
#      bash scripts/run_scales.sh --dataset IRDST --scales s --epochs 1 --batch 4
#
#  断点续跑（默认开启，不用加参数）——每个尺度开跑前自动判定状态：
#    已完成  = 训练跑满 epoch 且 eval/summary.csv 已存在  → 整个跳过
#    待评估  = 训练跑满 epoch，但评估缺失/失败            → 只补评估
#    中断    = 训练没跑满、但有 weights/last.pt           → 用 last.pt 断点续训
#    未开始  = 什么都没有                                  → 全新训练
#  所以中途 Ctrl-C / 掉线 / 单尺度失败，重新执行同一条命令即可接着跑。
#     --force     忽略已有产物，所有尺度全部重跑
#     --restart   中断的训练从头重训（默认是断点续训）
#
#  可用环境变量覆盖：
#    DATA_ROOT   数据集根目录（默认 /project/IDIP/Dataset/IR_SEQ）
#    PYTHON      指定解释器（默认直接用 PATH 里的 python / python3）
#    WEIGHTS_DIR 权重目录（默认 <仓库>/../weights）
#    LOG_DIR     日志目录（默认 <仓库>/logs_scales）
#
#  解释器：脚本不绑定任何虚拟环境路径。执行前请先 conda activate / source
#  好你要用的环境；脚本只会用 PATH 里的 python 做一次 `import ultralytics` 体检，
#  体检不过就报错退出（--list / --dry-run 时只警告）。
# =============================================================================
set -u -o pipefail

cd "$(dirname "$0")/.." || exit 1
CODE_ROOT="$(pwd)"

# ---------- 默认参数 ----------
DATASET=""
SCALES="s,m,l,x"
DEVICE="0"
EPOCHS="300"
IMGSZ="640"
BATCH="24"
DATA_ROOT="${DATA_ROOT:-/project/IDIP/Dataset/IR_SEQ}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$CODE_ROOT/../weights}"
LOG_DIR="${LOG_DIR:-$CODE_ROOT/logs_scales}"
EVAL_CONF="0.001"
EVAL_IOU="0.55"
PR_CONF="0.5"
DO_VIS=0
SKIP_TRAIN=0
SKIP_EVAL=0
DRY_RUN=0
LIST_ONLY=0
FORCE=0
RESTART=0

usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; }

# 一行打印整条命令（printf 会按参数个数重复格式串，所以拆成两步）
print_cmd() {
  printf '  [dry-run]'
  printf ' %q' "$@"
  echo
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --scales)
      SCALES="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --imgsz)
      IMGSZ="$2"
      shift 2
      ;;
    --batch)
      BATCH="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --weights-dir)
      WEIGHTS_DIR="$2"
      shift 2
      ;;
    --vis)
      DO_VIS=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --list)
      LIST_ONLY=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --restart)
      RESTART=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "[错误] 未知参数: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [ -z "$DATASET" ]; then
  echo "[错误] 必须指定 --dataset" >&2
  usage
  exit 2
fi

# ---------- 数据集注册表：新增数据集只需在这里加一段 ----------
# 相对路径都挂在 $DATA_ROOT 下，本地/服务器只是 DATA_ROOT 取值不同。
case "$DATASET" in
  IRDST)
    DS_YAML="$DATA_ROOT/YOLO_VER/IRDST_YOLO/dataset.yaml"
    DS_GT_FORMAT="coco"
    DS_COCO_GT="$DATA_ROOT/IRDST/IRDST_instances_test2017.json"
    DS_IMG_PATH="$DATA_ROOT/IRDST"
    DS_SPLIT="test"
    ;;
  DAUB)
    DS_YAML="$DATA_ROOT/YOLO_VER/DAUB_YOLO/dataset.yaml"
    DS_GT_FORMAT="coco"
    DS_COCO_GT="$DATA_ROOT/Tridos_Ver/DAUB/annotations/instances_test2017.json"
    DS_IMG_PATH="$DATA_ROOT/Tridos_Ver/DAUB"
    DS_SPLIT="test"
    ;;
  ITSDT | ITSDT_YOLO)
    DATASET="ITSDT"
    DS_YAML="$DATA_ROOT/YOLO_VER/ITSDT_YOLO/dataset.yaml"
    DS_GT_FORMAT="coco"
    DS_COCO_GT="$DATA_ROOT/ITSDT-15K/annotations/instances_test2017.json"
    DS_IMG_PATH="$DATA_ROOT/ITSDT-15K"
    DS_SPLIT="test"
    ;;
  MIST_det | MIST)
    DATASET="MIST_det"
    DS_YAML="$DATA_ROOT/YOLO_VER/MIST_YOLO_det/dataset.yaml"
    DS_GT_FORMAT="yolo"
    DS_COCO_GT=""
    DS_IMG_PATH=""
    DS_SPLIT="test"
    ;;
  DAUB-H | DAUB_H)
    DATASET="DAUB-H"
    DS_YAML="$DATA_ROOT/YOLO_VER/DAUB-H_YOLO/dataset.yaml"
    DS_GT_FORMAT="coco"
    DS_COCO_GT="$DATA_ROOT/DAUB-H/annotations/instances_test2017.json"
    DS_IMG_PATH="$DATA_ROOT/DAUB-H"
    DS_SPLIT="test"
    ;;
  IRDST-R | IRDST_R)
    DATASET="IRDST-R"
    DS_YAML="$DATA_ROOT/YOLO_VER/IRDST-R_YOLO/dataset.yaml"
    DS_GT_FORMAT="coco"
    DS_COCO_GT="$DATA_ROOT/IRDST-R/IRDST1_anns/instances_test2017.json"
    DS_IMG_PATH="$DATA_ROOT/IRDST-R"
    DS_SPLIT="test"
    ;;
  *)
    echo "[错误] 未知数据集 '$DATASET'。已支持：IRDST、DAUB、ITSDT-15K、MIST_det、DAUB-H、IRDST-R" >&2
    exit 2
    ;;
esac

# run 名后缀：MIST_det 这种本身已带 _det 的不要再拼一次，
# 最终 run 名形如 yolo11s_IRDST_det / yolo11s_MIST_det
case "$DATASET" in
  *_det) NAME_BASE="$DATASET" ;;
  *) NAME_BASE="${DATASET}_det" ;;
esac

# ---------- 解释器：用你当前激活的环境 ----------
# 这里不指定任何具体虚拟环境路径、也不会替你切换环境。
# 请先 conda activate / source 好环境再执行；临时覆盖用 PYTHON=/path/to/python。
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif command -v python > /dev/null 2>&1; then
  PY="python"
elif command -v python3 > /dev/null 2>&1; then
  PY="python3"
else
  echo "[错误] PATH 里找不到 python / python3。" >&2
  echo "       请先激活虚拟环境，或用 PYTHON=/path/to/python 指定。" >&2
  exit 3
fi

# 只做一次体检，不替你换环境：--list/--dry-run 下退化为警告
if ! "$PY" -c 'import ultralytics' > /dev/null 2>&1; then
  echo "[错误] 解释器 '$PY' 无法 import ultralytics，说明虚拟环境没激活或选错了。" >&2
  echo "       激活正确的环境后重试；临时覆盖用 PYTHON=/path/to/python。" >&2
  if [ "$LIST_ONLY" != "1" ] && [ "$DRY_RUN" != "1" ]; then
    exit 3
  fi
  echo "[警告] 当前是 --list/--dry-run，不执行训练/评估，继续。" >&2
fi

IFS=',' read -r -a SCALE_ARR <<< "$SCALES"
PROJECT="$CODE_ROOT/runs/detect"
mkdir -p "$LOG_DIR" 2> /dev/null

echo "======================================================================"
echo " 数据集        : $DATASET"
echo " 尺度          : ${SCALE_ARR[*]}"
echo " 代码根        : $CODE_ROOT"
echo " 数据集根      : $DATA_ROOT"
echo " dataset.yaml  : $DS_YAML"
echo " GT 形式       : $DS_GT_FORMAT"
[ -n "$DS_COCO_GT" ] && echo " COCO GT       : $DS_COCO_GT"
echo " 权重目录      : $WEIGHTS_DIR"
echo " 解释器        : $PY"
echo " device/epochs/batch : $DEVICE / $EPOCHS / $BATCH   (imgsz=$IMGSZ)"
if [ "$DO_VIS" = "1" ]; then echo " 评估 vis      : 生成"; else echo " 评估 vis      : 不生成(--no-vis)"; fi
echo " 输出项目目录  : $PROJECT"
echo "======================================================================"

if [ ! -f "$DS_YAML" ]; then
  echo "[警告] 找不到 dataset.yaml: $DS_YAML" >&2
fi
if [ -n "$DS_COCO_GT" ] && [ ! -f "$DS_COCO_GT" ]; then
  echo "[警告] 找不到 COCO GT: $DS_COCO_GT" >&2
fi

# ===========================================================================
#  断点续跑：状态判定
# ===========================================================================
# 一个尺度的状态由「训练是否跑完」+「评估是否产出」共同决定：
#   done    训练跑满 epoch 且 eval/summary.csv 已存在  → 整个跳过
#   trained 训练跑满 epoch，但评估缺失/失败            → 只补评估
#   partial 训练中断（results.csv 不足 epoch，有 last.pt）→ 断点续训
#   none    什么都没跑                                  → 全新训练
completed_epochs() { # $1 = run 目录
  local csv="$1/results.csv"
  [ -f "$csv" ] || {
    echo 0
    return
  }
  awk -F, 'NR>1 { n++; if ($1+1 > m) m = $1+1 } END { print (m > n ? m : n) + 0 }' "$csv"
}

target_epochs() { # $1 = run 目录；优先用该 run 自己 args.yaml 里的 epochs
  local y="$1/args.yaml" e
  if [ -f "$y" ]; then
    e="$(sed -n 's/^epochs:[[:space:]]*//p' "$y" | head -1)"
    case "$e" in
      '' | *[!0-9]*) ;;
      *)
        echo "$e"
        return
        ;;
    esac
  fi
  echo "$EPOCHS"
}

resolve_run_dir() { # $1 = run 名；兼容历史的两层 project 路径
  local name="$1" cand f
  for cand in "$PROJECT/$name" "$PROJECT/runs/detect/$name" "$CODE_ROOT/runs/detect/$name"; do
    [ -d "$cand" ] && {
      echo "$cand"
      return
    }
  done
  f="$(find "$PROJECT" "$CODE_ROOT/runs" -maxdepth 4 -type d -name "$name" 2> /dev/null | head -1)"
  if [ -n "$f" ]; then echo "$f"; else echo "$PROJECT/$name"; fi
}

scale_state() { # $1 = run 目录
  local rd="$1" done_ep target
  [ -d "$rd" ] || {
    echo none
    return
  }
  done_ep="$(completed_epochs "$rd")"
  target="$(target_epochs "$rd")"
  if [ "$done_ep" -ge "$target" ]; then
    if [ -f "$rd/eval/summary.csv" ]; then echo done; else echo trained; fi
    return
  fi
  if [ -f "$rd/weights/last.pt" ]; then
    echo partial
    return
  fi
  echo none
}

state_cn() {
  local rd="$1" st
  st="$(scale_state "$rd")"
  case "$st" in
    done) echo "已完成(训练+评估) → 跳过" ;;
    trained) echo "训练已完成($(target_epochs "$rd") epoch)，评估缺失 → 只补评估" ;;
    partial) echo "训练中断 $(completed_epochs "$rd")/$(target_epochs "$rd") epoch → 断点续训" ;;
    *) echo "未开始 → 全新训练" ;;
  esac
}

# 开跑前的状态预览
echo
echo "---------------- 各尺度断点状态 ----------------"
for s in "${SCALE_ARR[@]}"; do
  nm="yolo11${s}_${NAME_BASE}"
  printf '  %-10s %-26s %s\n' "yolo11$s" "$nm" "$(state_cn "$(resolve_run_dir "$nm")")"
done
echo "------------------------------------------------"
if [ "$FORCE" = "1" ]; then echo "  (--force：忽略上述状态，全部重跑)"; fi
if [ "$RESTART" = "1" ]; then echo "  (--restart：中断的训练从头开始，不做续训)"; fi

if [ "$LIST_ONLY" = "1" ]; then
  echo
  echo "（--list：仅预览，不执行）"
  exit 0
fi

run_one() {
  local s="$1"
  local model="$WEIGHTS_DIR/yolo11$s.pt"
  local name="yolo11${s}_${NAME_BASE}"
  local run_dir
  local log="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"
  local rc=0 erc=0 st=""
  local do_train=1 use_resume=0

  echo
  echo "######################################################################"
  echo "#  尺度 yolo11$s   →  $name"
  echo "######################################################################"

  if [ ! -f "$model" ]; then
    echo "[错误] 缺少权重: $model —— 跳过 yolo11$s" >&2
    return 1
  fi

  # ---------------- 断点状态判定 ----------------
  run_dir="$(resolve_run_dir "$name")"
  st="$(scale_state "$run_dir")"
  echo "[$name] run 目录 : $run_dir"
  echo "[$name] 断点状态 : $(state_cn "$run_dir")"
  if [ "$FORCE" = "1" ]; then
    st="none"
    echo "[$name] --force：忽略已有产物，重新训练"
  fi
  case "$st" in
    done)
      echo "[$name] 训练与评估均已完成，跳过 ✅"
      return 0
      ;;
    trained)
      do_train=0
      echo "[$name] 跳过训练，仅补做评估"
      ;;
    partial)
      if [ "$RESTART" = "1" ]; then
        echo "[$name] 训练中断，--restart：从头重训（原产物会被覆盖）"
      else
        use_resume=1
        echo "[$name] 训练中断，用 weights/last.pt 断点续训（加 --restart 可改为重训）"
      fi
      ;;
    *) : ;;
  esac
  [ "$SKIP_TRAIN" = "1" ] && do_train=0

  # ---------------- 训练 ----------------
  if [ "$do_train" = "1" ]; then
    local train_model="$model" train_extra=()
    if [ "$use_resume" = "1" ] && [ -f "$run_dir/weights/last.pt" ]; then
      train_model="$run_dir/weights/last.pt"
      train_extra=(--resume)
    fi
    echo "[$name] 开始训练 ...  日志: $log"
    if [ "$DRY_RUN" = "1" ]; then
      print_cmd "$PY" train_det.py \
        --data "$DS_YAML" --model "$train_model" --name "$name" --project "$PROJECT" \
        --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" --device "$DEVICE" \
        ${train_extra[@]+"${train_extra[@]}"}
    else
      "$PY" train_det.py \
        --data "$DS_YAML" \
        --model "$train_model" \
        --name "$name" \
        --project "$PROJECT" \
        --epochs "$EPOCHS" \
        --imgsz "$IMGSZ" \
        --batch "$BATCH" \
        --device "$DEVICE" \
        ${train_extra[@]+"${train_extra[@]}"} 2>&1 | tee "$log"
      rc=${PIPESTATUS[0]}
      # 续训失败时自动回退成从头训练，免得把整个尺度卡死
      if [ "$rc" -ne 0 ] && [ "$use_resume" = "1" ]; then
        echo "[警告] 断点续训失败(exit=$rc)，自动回退为从头训练 $name" >&2
        "$PY" train_det.py \
          --data "$DS_YAML" \
          --model "$model" \
          --name "$name" \
          --project "$PROJECT" \
          --epochs "$EPOCHS" \
          --imgsz "$IMGSZ" \
          --batch "$BATCH" \
          --device "$DEVICE" 2>&1 | tee -a "$log"
        rc=${PIPESTATUS[0]}
      fi
      if [ "$rc" -ne 0 ]; then
        echo "[错误] 训练失败(exit=$rc)，跳过 $name 的评估" >&2
        return "$rc"
      fi
    fi
  fi

  # ---------------- 训练后重新定位 run 目录 ----------------
  run_dir="$(resolve_run_dir "$name")"
  echo "[$name] run 目录: $run_dir"

  # ---------------- 评估 ----------------
  if [ "$SKIP_EVAL" != "1" ]; then
    if [ ! -f "$run_dir/weights/best.pt" ] && [ "$DRY_RUN" != "1" ]; then
      echo "[错误] 找不到权重 $run_dir/weights/best.pt，跳过评估" >&2
      return 1
    fi
    local ecmd=("$PY" eval/eval_det.py
      --gt-format "$DS_GT_FORMAT"
      --model "$run_dir/weights/best.pt"
      --save-path "$run_dir/eval"
      --imgsz "$IMGSZ"
      --conf "$EVAL_CONF"
      --iou "$EVAL_IOU"
      --pr_conf "$PR_CONF"
      --device "$DEVICE")
    if [ "$DS_GT_FORMAT" = "coco" ]; then
      ecmd+=(--coco-gt "$DS_COCO_GT" --dataset-img-path "$DS_IMG_PATH")
    else
      ecmd+=(--data "$DS_YAML" --split "$DS_SPLIT")
    fi
    [ "$DO_VIS" = "1" ] || ecmd+=(--no-vis)

    echo "[$name] 开始评估 ..."
    if [ "$DRY_RUN" = "1" ]; then
      print_cmd PYTHONPATH=. "${ecmd[@]}"
    else
      # 子 shell 里切到 CODE_ROOT，保证 `import ultralytics` 命中本仓库
      (cd "$CODE_ROOT" && PYTHONPATH=. "${ecmd[@]}") 2>&1 \
        | tee -a "$log" | tail -n 22
      erc=${PIPESTATUS[0]}
      if [ "$erc" -ne 0 ]; then
        echo "[错误] 评估失败(exit=$erc): $name" >&2
        return "$erc"
      fi
    fi
  fi
  echo "[$name] 完成"
}

FAILED=()
for s in "${SCALE_ARR[@]}"; do
  if ! run_one "$s"; then
    FAILED+=("$s")
    echo "[警告] yolo11$s 未成功，继续下一个尺度" >&2
  fi
done

# ---------------- 汇总（summary.csv 列序见 eval_det.py COCO_SUMMARY_FIELDS） ----------------
echo
echo "======================================================================"
echo " 汇总：$DATASET  各尺度指标（TDCNet 口径）"
echo "======================================================================"
printf '%-10s %-10s %-8s %-9s %-8s %-8s %-8s\n' "scale" "images" "AP50" "AP_small" "P50" "R50" "F1_50"
for s in "${SCALE_ARR[@]}"; do
  csv="$PROJECT/yolo11${s}_${NAME_BASE}/eval/summary.csv"
  if [ ! -f "$csv" ]; then
    csv="$(find "$PROJECT" -maxdepth 4 -path "*yolo11${s}_${NAME_BASE}/eval/summary.csv" 2> /dev/null | head -1)"
  fi
  if [ -n "$csv" ] && [ -f "$csv" ]; then
    awk -F, -v s="$s" 'NR==2 {printf "%-10s %-10s %-8.4f %-9.4f %-8.4f %-8.4f %-8.4f\n", s, $2, $4, $6, $12, $13, $14}' "$csv"
  else
    printf '%-10s %s\n' "yolo11$s" "（没有 summary.csv）"
  fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo
  echo "[警告] 以下尺度未成功: ${FAILED[*]}"
  exit 1
fi
echo
echo "全部完成 ✅   日志目录: $LOG_DIR"
