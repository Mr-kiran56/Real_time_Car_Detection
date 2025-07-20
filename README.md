

## 🚗 Real-Time Car Detection, Tracking, and Counting with YOLOv8 + SORT

### 📌 **Project Summary:**

This project is a real-time car detection and counting system built using the **YOLOv8 object detection model** by Ultralytics and the **SORT tracking algorithm**. It processes a traffic video feed, tracks each car with a unique ID, and counts them **only once** as they cross a predefined line — ensuring accuracy and preventing duplicate counts.

---

### 🧠 **Key Technologies Used:**

* **YOLOv8** – for object detection with high speed and precision
* **SORT (Simple Online and Realtime Tracking)** – for assigning and maintaining unique IDs across frames
* **OpenCV (cv2)** – for video processing and frame manipulation
* **cvzone** – to enhance visual representation with stylized bounding boxes and text
* **NumPy** – for efficient numerical operations on detection arrays

---

### 🎯 **Core Features:**

✅ **Car-Only Detection**
Filters YOLOv8 predictions to include only the `"car"` class with a confidence score above 0.35.

✅ **Unique ID Tracking with SORT**
Uses SORT to assign a consistent ID to each detected car, ensuring accurate tracking across multiple frames.

✅ **Cross-Line Counting Logic**
Defines a virtual horizontal line across the frame. A car is counted only when its center crosses this line — with a buffer — and hasn’t been counted before.

✅ **Live Visual Feedback**
Draws bounding boxes, IDs, confidence scores, and a counting line directly on the video using `cvzone`.

✅ **Duplicate-Free Count**
Maintains a Python set (`counted_ids`) to ensure cars are **only counted once**, no matter how long they stay in view.

---

### 🎥 **How It Works:**

1. Load video using OpenCV from a local file.
2. Resize frames to a fixed resolution for consistency.
3. Apply YOLOv8 to each frame to detect cars.
4. Update car positions and assign tracking IDs using SORT.
5. Draw a red horizontal line — when a car’s center crosses it and hasn't been counted before, increment the total.
6. Display all visual data on the video feed in real-time.

---

### 📊 **Potential Applications:**

* 🚦 **Smart Traffic Monitoring Systems**
* 📈 **Vehicle Flow Analysis for Urban Planning**
* 📷 **Intelligent Surveillance and Toll Systems**
* 📚 **Research Projects on Object Detection and Tracking**
* 🧠 **Model Benchmarking for ML & CV Engineers**

---

### 🛠️ **Next Possible Enhancements:**

* Display **real-time total car count** on the screen
* Count **multiple vehicle types** like trucks, buses, bikes
* Log all detections and timestamps to a **CSV/Database**
* Integrate with **Flask/Django** for web-based monitoring
* Add a **UI dashboard** for reports and graph visualizations

---
