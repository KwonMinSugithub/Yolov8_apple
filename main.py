from ultralytics import YOLO
import cv2, cvzone
import math

model = YOLO(
    r"C:\Users\user\Desktop\apples.v2i.yolov8\runs\detect\apple\weights\best.pt"
)

classNames = ["fresh", "rotten"]
classColors = [(0, 255, 0), (0, 0, 255)]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 바운딩 박스
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cls = int(box.cls[0])
            color = classColors[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cvzone.cornerRect(frame, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100  # 정확도
            cls = int(box.cls[0])
            cvzone.putTextRect(
                frame,
                f"{classNames[cls]} {conf}",
                (max(0, x1), max(35, y1)),
                scale=1,
                thickness=1,
            )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
