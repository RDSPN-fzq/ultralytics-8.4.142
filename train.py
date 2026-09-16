import os
from pathlib import Path

from ultralytics import YOLO


def main():
    root = Path(r"C:\Users\Administrator\Desktop\ultralytics-main")
    os.chdir(root)

    model = YOLO(str(root / "yolov8s.pt"))

    model.train(
        data=str(root / "data" / "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        project=str(root / "runs" / "detect"),
        name="pcb_train",
        patience=30,
        seed=0,
        plots=True,
    )


# Windows 下必须保留主入口保护。
if __name__ == "__main__":
    main()