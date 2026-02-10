from ultralytics import YOLO

def train():
    model = YOLO(
        "yolov8n-cls.pt"
    )

    model.train(
        data="dataset",
        epochs=50,
        imgsz=224,
        batch=8,
        project="runs/classify",
        name="rice_guard"
    )

if __name__ == "__main__":
    train()
