from pathlib import Path

from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parent
    model = YOLO(str(root / "runs/detect/pcb_train/weights/best.pt"))

    results = model.predict(
        source=str(root / "data/image_predict"),
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device=0,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(root / "runs/detect"),
        name="pcb_predict",
        stream=True,
    )

    for result in results:
        print(f"{Path(result.path).name}: {len(result.boxes)} objects")


if __name__ == "__main__":
    main()
