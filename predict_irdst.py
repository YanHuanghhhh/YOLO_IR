# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Inference with trained IRDST YOLO model.

Usage:
    python predict_irdst.py                                                           # predict all test images
    python predict_irdst.py --source path/to/image.jpg                                # single image
    python predict_irdst.py --source path/to/folder --conf 0.5 --save-txt             # folder with custom conf
    python predict_irdst.py --source path/to/video.mp4 --device 0 --show              # video
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="IRDST YOLO inference")
    parser.add_argument(
        "--model",
        type=str,
        default="E:/Work/CV/Example/56_YOLO/runs/detect/runs/irdst/yolo26n_irdst/weights/best.pt",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="E:/Work/Dataset/infrared_sequence/YOLO_VER/IRDST_YOLO/test/images",
        help="Input source: image path, folder, or video",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0", help="GPU device, e.g. '0' or 'cpu'")
    parser.add_argument("--save-txt", action="store_true", help="Save results as .txt labels")
    parser.add_argument("--save-conf", action="store_true", help="Save confidence scores in label files")
    parser.add_argument("--show", action="store_true", help="Display results in a window")
    parser.add_argument(
        "--project",
        type=str,
        default="runs/irdst",
        help="Output project directory",
    )
    parser.add_argument("--name", type=str, default="predict", help="Output run name")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing output")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show=args.show,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
