# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train YOLO semantic/instance segmentation on infrared datasets (NUDT-MIRSDT-v2, etc.).

Usage:
    # Semantic segmentation (推荐)
    python train_seg.py --data E:/.../NUDT_YOLO/dataset.yaml --model ../weights/yolo26n-sem.pt
    python train_seg.py --data E:/.../NUDT_YOLO/dataset.yaml --model ../weights/yolo26s-sem.pt

    # Instance segmentation
    python train_seg.py --data E:/.../NUDT_YOLO/dataset.yaml --model ../weights/yolo26n-seg.pt
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO segmentation on infrared datasets")
    parser.add_argument(
        "--data",
        type=str,
        default="E:/Work/Dataset/infrared_sequence/YOLO_VER/NUDT_YOLO/dataset.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../weights/yolo26n-sem.pt",
        help="Model weights (yolo26n-sem.pt / yolo26s-sem.pt / yolo26n-seg.pt / yolo26s-seg.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--device", type=str, default="0", help="GPU device(s), e.g. '0' or '0,1'")
    parser.add_argument("--name", type=str, default=None, help="Output run name (auto-generated if not set)")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    # Auto-generate name from model filename
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
        project="runs/seg",
        name=name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
