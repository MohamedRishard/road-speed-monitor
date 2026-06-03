# src/detector.py
"""
Vehicle Detection Module
Uses YOLOv8 to detect vehicles in video frames.
Filters detections to vehicle classes only (car, motorcycle, bus, truck).
"""

from ultralytics import YOLO
import cv2
import numpy as np
import sys
import os

# Add project root to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import YOLO_MODEL, CONFIDENCE, VEHICLE_CLASSES


class VehicleDetector:
    """
    Wraps YOLOv8 model for vehicle-only detection.
    
    Why a class?
    - Loads the model once and reuses it across frames
    - Loading a model is expensive; doing it per-frame would be very slow
    """

    def __init__(self):
        print(f"[Detector] Loading YOLOv8 model: {YOLO_MODEL}")
        self.model = YOLO(YOLO_MODEL)
        self.confidence = CONFIDENCE
        self.vehicle_classes = VEHICLE_CLASSES
        print(f"[Detector] Model loaded. Confidence threshold: {self.confidence}")
        print(f"[Detector] Tracking classes: {self.vehicle_classes} "
              f"(2=car, 3=motorcycle, 5=bus, 7=truck)")

    def detect(self, frame):
        """
        Run detection on a single frame.

        Args:
            frame: BGR image (numpy array from OpenCV)

        Returns:
            List of detections, each as a dict:
            {
                'bbox': [x1, y1, x2, y2],   # bounding box corners
                'confidence': float,          # detection confidence
                'class_id': int,              # COCO class ID
                'class_name': str             # human-readable label
            }
        """
        results = self.model(
            frame,
            conf=self.confidence,
            classes=self.vehicle_classes,  # only detect vehicles
            verbose=False                   # suppress per-frame console output
        )

        detections = []

        # results[0] because we pass one frame at a time
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.model.names[class_id]

            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': round(conf, 2),
                'class_id': class_id,
                'class_name': class_name
            })

        return detections


def draw_detections(frame, detections):
    """
    Draw bounding boxes and labels on a frame.

    Args:
        frame: BGR image (numpy array)
        detections: list of detection dicts from VehicleDetector.detect()

    Returns:
        Annotated frame
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class_name']} {det['confidence']}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label background for readability
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)

        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return frame


def test_detector(video_path):
    """
    Quick test function — runs detector on a video and displays results.
    Press 'q' to quit, SPACE to pause.

    Args:
        video_path: path to video file
    """
    print(f"\n[Test] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    detector = VehicleDetector()

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Test] Video info: {width}x{height} @ {fps:.1f}fps | {total_frames} frames")
    print("[Test] Press 'q' to quit | SPACE to pause\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[Test] Video ended.")
            break

        frame_count += 1

        # Run detection
        detections = detector.detect(frame)

        # Draw results
        annotated = draw_detections(frame.copy(), detections)

        # Show frame number and detection count
        info = f"Frame: {frame_count}/{total_frames} | Vehicles: {len(detections)}"
        cv2.putText(annotated, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Vehicle Detection Test", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[Test] Quit by user.")
            break
        elif key == ord(' '):
            cv2.waitKey(0)  # pause until SPACE again

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[Test] Processed {frame_count} frames.")
    print(f"[Test] Detection test complete.")


# ── Run directly to test ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/detector.py data/videos/traffic.mp4")
    else:
        test_detector(sys.argv[1])