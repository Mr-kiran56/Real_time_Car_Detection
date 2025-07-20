from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
from sort import *

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')

# Open the video file
cap = cv2.VideoCapture('newaniaml vedio edited - Made with Clipchamp.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Initialize animal count and ID tracking
total_counts = {'elephant': 0, 'horse': 0, 'zebra': 0, 'cow': 0, 'bear': 0}
counted_ids = set()
animal_classes = {'horse', 'cow', 'elephant', 'bear', 'zebra'}

classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',  'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
# SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1680, 920))
    results = model(frame)

    detections = np.empty((0, 5))
    animal_info = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = classNames[cls]

            if class_name in animal_classes and conf > 0.60:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                animal_info.append((x1, y1, x2, y2, class_name, conf))

    # Tracker update
    results_tracker = tracker.update(detections)

    for track, (x1, y1, x2, y2, class_name, conf) in zip(results_tracker, animal_info):
        track_id = int(track[4])
        w, h = x2 - x1, y2 - y1

        if track_id not in counted_ids:
            counted_ids.add(track_id)
            if class_name in total_counts:
                total_counts[class_name] += 1

        # Draw only for tracked animals
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2)
        cvzone.putTextRect(frame, f'{class_name} ID:{track_id}', (x1, y1 - 10),
                           scale=1.2, thickness=2, colorR=(0, 0, 0), colorT=(255, 255, 255))

    # Show total counts on top-left
    y_offset = 30
    for animal, count in total_counts.items():
        if count > 0:
            cvzone.putTextRect(frame, f'{animal.title()}: {count}', (30, y_offset),
                               scale=1.2, thickness=2, colorR=(0, 0, 0), colorT=(255, 255, 255))
            y_offset += 40

    cv2.imshow("Animal Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
