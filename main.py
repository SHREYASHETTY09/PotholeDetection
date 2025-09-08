# main.py
# Simple pothole detection on video using OpenCV (no training needed)

import cv2
import numpy as np
import os

# 1) ---- CHANGE THIS TO YOUR EXACT FILE NAME (mp4/avi/etc.) ----
VIDEO_PATH = r"C:\Users\HP\OneDrive\Desktop\PotholeDetection\pothole.mp4"
# ---------------------------------------------------------------

FOLDER = os.path.dirname(VIDEO_PATH)
OUTPUT_PATH = os.path.join(FOLDER, "pothole_output.mp4")

def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at: {path}")
    # Get video properties (fallbacks if FPS is unknown)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, w, h, fps

def make_writer(path, w, h, fps):
    # mp4 writer; if it fails on your PC, try .avi with 'XVID' (see note below)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

def detect_potholes(frame, roi_top_ratio=0.45):
    """
    Very simple heuristic:
      - Look only at bottom part of the frame (road region)
      - Enhance contrast, find edges and blobs
      - Keep dark, irregular shapes of a reasonable size
    """
    h, w = frame.shape[:2]
    roi_y = int(h * roi_top_ratio)
    roi = frame[roi_y:, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Contrast Limited Adaptive Histogram Equalization (helps on dark roads)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Smooth + edges
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)

    # Strengthen edges and close small gaps
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find candidate regions
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pothole_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Ignore too small/big areas (tune if needed)
        if area < 400 or area > 30000:
            continue

        x, y, w_box, h_box = cv2.boundingRect(cnt)
        if w_box < 20 or h_box < 20:
            continue

        aspect = w_box / float(h_box)
        if aspect > 4.5 or aspect < 0.25:
            continue

        # Solidity (irregular shapes are common for potholes)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = area / hull_area
        if solidity > 0.95:
            continue

        # Check darkness inside the contour
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_intensity = cv2.mean(gray_eq, mask=mask)[0]
        if mean_intensity > 140:  # too bright → likely not a pothole
            continue

        # Convert ROI coords back to full-frame coords
        pothole_boxes.append((x, y + roi_y, w_box, h_box))

    return pothole_boxes, edges

def main():
    cap, w, h, fps = open_video(VIDEO_PATH)
    writer = make_writer(OUTPUT_PATH, w, h, fps)

    print(f"Reading: {VIDEO_PATH}")
    print(f"Saving:  {OUTPUT_PATH} (press 'q' to stop preview)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        boxes, debug_edges = detect_potholes(frame, roi_top_ratio=0.45)

        # Draw results
        for (x, y, bw, bh) in boxes:
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(frame, f"Potholes: {len(boxes)}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        writer.write(frame)

        # Show live preview (close if it bothers you)
        cv2.imshow("Pothole Detection", frame)
        cv2.imshow("Edges (debug)", debug_edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
