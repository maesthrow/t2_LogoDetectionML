from ultralytics import YOLO


yolo_data = ".\\data\\data.yaml"


def train_yolo_model():
    # Load a model
    model = YOLO("yolo11l-seg.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=yolo_data, epochs=400, imgsz=640, batch=-1)
    print("SUCCESS!")


