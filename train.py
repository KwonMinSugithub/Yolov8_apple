from ultralytics import YOLO

# 모델 로드
model = YOLO("Yolo-Weights/yolov8n.pt")
# 모델 훈련
results = model.train(
    data=r"C:\Users\user\Desktop\apples.v2i.yolov8\data.yaml",
    imgsz=640,
    epochs=50,
    batch=8,
    name="apple",
)
