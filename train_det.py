# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train YOLO detection on infrared datasets.

Usage:
    python train_det.py                                          # use default params
    python train_det.py --epochs 200 --batch 32 --device 0       # override training params
    python train_det.py --data path/to/dataset.yaml --model path/to/weights.pt --name my_run
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO detection on infrared datasets")
    parser.add_argument(
        "--data",
        type=str,
        default="E:/Work/Dataset/infrared_sequence/YOLO_VER/IRDST_YOLO/dataset.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument("--model", type=str, default="../weights/yolo26n.pt", help="Path to model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--device", type=str, default="0", help="GPU device(s), e.g. '0' or '0,1'")
    parser.add_argument("--name", type=str, default=None, help="Output run name (auto-generated if not set)")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output project directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续训：--model 需指向某个 run 的 weights/last.pt，"
        "其余超参以该 checkpoint 内保存的为准（epochs/lr/增强等），"
        "仅 device/batch/imgsz/workers 允许用命令行覆盖",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # —— 断点续训：直接接上次的 last.pt 继续，不重新初始化 ——
    # Ultralytics 的 resume 语义（见 ultralytics/engine/trainer.py: check_resume）：
    #   超参从 checkpoint 里恢复，避免"续训却换了配方"；
    #   只有 device/batch/imgsz/workers/patience 等少数项允许用命令行覆盖，
    #   所以这里显式传 device/batch/imgsz，保证续训仍跑在你指定的那张卡上。
    if args.resume:
        print(f"[INFO] 断点续训: {args.model} （epochs 等超参取自 checkpoint）")
        model = YOLO(args.model)
        model.train(resume=True, device=args.device, batch=args.batch, imgsz=args.imgsz, workers=4)
        return

    model = YOLO(args.model)

    # Auto-generate name from model filename if not set
    name = args.name
    if name is None:
        stem = args.model.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].replace(".pt", "")
        dataset = args.data.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].replace(".yaml", "").replace("_YOLO", "").lower()
        name = f"{stem}_{dataset}"

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        optimizer="AdamW",
        lr0=1e-3,
        momentum=0.9,
        weight_decay=0.025,
        warmup_epochs=3,
        cos_lr=True,
        close_mosaic=15,
        project=args.project,
        name=name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
