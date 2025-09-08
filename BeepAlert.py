from ultralytics import YOLO
import cv2
import math
import winsound   # ✅ for beep alerts on Windows

# Load your trained YOLO model
model = YOLO(r"C:\Users\HP\OneDrive\Desktop\PotholeDetection\best1.pt")

# Input video path
video_path = r"C:\Users\HP\OneDrive\Desktop\PotholeDetection\pothole.mp4"
cap = cv2.VideoCapture(video_path)

# ---------------- Tracking Setup ---------------- #
next_id = 0
trackers = {}   # pothole_id -> centroid
pothole_info = {}  # pothole_id -> (area, severity, risk, color)
alerted_ids = set()  # ✅ track potholes already alerted

def get_centroid(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def classify_area(area):
    if area < 100000:
        return "LOW", "SAFE", (0, 255, 0)
    else:
        return "HIGH", "RISKY", (0, 0, 255)

def assign_id(centroid, area):
    global next_id

    # Find nearest existing tracker (centroid matching)
    for tid, prev_centroid in trackers.items():
        if math.dist(centroid, prev_centroid) < 50:  # same pothole
            trackers[tid] = centroid
            return tid

    # New pothole → assign new ID
    tid = next_id
    next_id += 1
    trackers[tid] = centroid

    # Lock area & severity at first detection
    severity, risk, color = classify_area(area)
    pothole_info[tid] = (area, severity, risk, color)

    return tid

# ---------------- Main Loop ---------------- #
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=640, verbose=False)
    annotated = frame.copy()

    high_risk_detected = False  # ✅ track if any high-risk pothole exists in frame

    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        area = (x2 - x1) * (y2 - y1)

        # Get centroid and pothole ID
        centroid = get_centroid(x1, y1, x2, y2)
        pothole_id = assign_id(centroid, area)

        # Use locked severity info
        locked_area, severity, risk, color = pothole_info[pothole_id]

        # ✅ Beep if pothole is HIGH-risk and not alerted before
        if severity == "HIGH":
            high_risk_detected = True
            if pothole_id not in alerted_ids:
                winsound.Beep(1200, 400)  # frequency=1200Hz, duration=400ms
                alerted_ids.add(pothole_id)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Draw label (locked area and severity)
        label = f"ID:{pothole_id} | Area:{locked_area} | {severity} | {risk}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - text_h - 6), (x1 + text_w, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ✅ Show big warning text if any HIGH-risk pothole is detected
    if high_risk_detected:
        cv2.putText(annotated, "HIGH-RISK POTHOLE AHEAD",
                    (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

    # Show detections live
    cv2.imshow("Pothole Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
