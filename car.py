from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
from sort import *
import time

model = YOLO('yolov8n.pt')


cap = cv2.VideoCapture('pro-2/videoplayback (1).mp4')


limits = [200, 450, 1200, 450]


total_cars = 0


counted_ids = set()


classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    frame = cv2.resize(frame, (1680, 920))


    results = model(frame)
    detections = np.empty((0, 5))


    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            currentClass = classNames[cls]


            if currentClass == "car" and conf > 0.35:
                cvzone.putTextRect(frame, f'{currentClass} {conf}', (x1, max(35, y1)), scale=1.5, thickness=2)
                cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2)

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    results_tracker = tracker.update(detections)


    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)


    for result in results_tracker:
        x1, y1, x2, y2, track_id = map(int, result)
        w, h = x2 - x1, y2 - y1


        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2)
        cvzone.putTextRect(frame, f'{int(track_id)}', (x1, max(35, y1)), scale=1.5, thickness=2)


        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 233, 0), cv2.FILLED)


        if limits[0] < cx < limits[2] and (limits[1] - 20) < cy < (limits[1] + 20):
            if track_id not in counted_ids:
                counted_ids.add(track_id)
                total_cars += 1
                # cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 4)
    # cvzone.putTextRect(frame, f'Total Cars: {total_cars}', (50, 50), scale=2, thickness=3, colorR=(0, 0, 0), colorT=(255, 255, 255))


    cv2.imshow("YOLOv8 Car Tracking", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
