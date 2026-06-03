# src/tracker.py
"""
Vehicle Tracking Module
Integrates ByteTrack (via Ultralytics) to assign persistent IDs to vehicles.

Why ByteTrack?
- State of the art multi-object tracker
- Built into Ultralytics — no separate installation
- Handles occlusions and re-identification well
- Fast enough for real-time use
"""

from ultralytics import YOLO
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    YOLO_MODEL, CONFIDENCE, VEHICLE_CLASSES,
    NORMAL_COLOR, OVERSPEED_COLOR, FONT_SCALE, BOX_THICKNESS
)


class VehicleTracker:
    """
    Combines YOLOv8 detection with ByteTrack tracking.

    Key difference from VehicleDetector:
    - Uses model.track() instead of model()
    - Each detection now has a persistent track_id
    - Same vehicle keeps the same ID across frames
    """

    def __init__(self):
        print(f"[Tracker] Loading YOLOv8 + ByteTrack...")
        self.model = YOLO(YOLO_MODEL)
        self.confidence = CONFIDENCE
        self.vehicle_classes = VEHICLE_CLASSES
        print(f"[Tracker] Ready.")

    def track(self, frame):
        """
        Run detection + tracking on a single frame.

        Args:
            frame: BGR image (numpy array from OpenCV)

        Returns:
            List of tracked vehicles, each as a dict:
            {
                'track_id': int,              # persistent ID across frames
                'bbox': [x1, y1, x2, y2],    # bounding box
                'confidence': float,
                'class_id': int,
                'class_name': str,
                'center': (cx, cy)            # center point of bbox
            }
        """
        results = self.model.track(
            frame,
            conf=self.confidence,
            classes=self.vehicle_classes,
            tracker="bytetrack.yaml",   # use ByteTrack algorithm
            persist=True,               # CRITICAL: maintains track IDs between frames
            verbose=False
        )

        tracked_vehicles = []

        if results[0].boxes.id is None:
            # No vehicles tracked this frame
            return tracked_vehicles

        boxes = results[0].boxes

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            track_id  = int(boxes.id[i].cpu().numpy())
            conf      = float(boxes.conf[i].cpu().numpy())
            class_id  = int(boxes.cls[i].cpu().numpy())
            class_name = self.model.names[class_id]

            # Calculate center point — used for speed estimation later
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            tracked_vehicles.append({
                'track_id':   track_id,
                'bbox':       [x1, y1, x2, y2],
                'confidence': round(conf, 2),
                'class_id':   class_id,
                'class_name': class_name,
                'center':     (cx, cy)
            })

        return tracked_vehicles


def draw_tracked_vehicles(frame, tracked_vehicles, speeds=None, speed_limit=60):
    """
    Draw bounding boxes with track IDs and optional speeds.

    Args:
        frame:            BGR image
        tracked_vehicles: list from VehicleTracker.track()
        speeds:           dict of {track_id: speed_kmh} (optional)
        speed_limit:      threshold for red highlighting

    Returns:
        Annotated frame
    """
    for v in tracked_vehicles:
        x1, y1, x2, y2 = v['bbox']
        track_id   = v['track_id']
        class_name = v['class_name']

        # Determine color based on speed
        color = NORMAL_COLOR  # green by default
        speed_text = ""

        if speeds and track_id in speeds:
            spd = speeds[track_id]
            speed_text = f" {spd:.1f}km/h"
            if spd > speed_limit:
                color = OVERSPEED_COLOR  # red for overspeeding

        label = f"ID:{track_id} {class_name}{speed_text}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Label background
        (w, h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
        cv2.rectangle(frame, (x1, y1 - 22), (x1 + w + 4, y1), color, -1)

        # Label text
        cv2.putText(frame, label, (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 1)

        # Center dot — useful for speed line visualization
        cv2.circle(frame, v['center'], 4, color, -1)

    return frame


def test_tracker(video_path):
    """
    Test tracking — watch IDs stay consistent across frames.
    Press 'q' to quit, SPACE to pause.
    """
    print(f"\n[Test] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    tracker = VehicleTracker()

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Test] Video: {width}x{height} @ {fps:.1f}fps | {total_frames} frames")
    print("[Test] Watch the IDs — they should stay consistent per vehicle.")
    print("[Test] Press 'q' to quit | SPACE to pause\n")

    frame_count   = 0
    all_track_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Test] Video ended.")
            break

        frame_count += 1
        tracked = tracker.track(frame)

        # Collect unique IDs seen so far
        for v in tracked:
            all_track_ids.add(v['track_id'])

        annotated = draw_tracked_vehicles(frame.copy(), tracked)

        # Info overlay
        info = (f"Frame: {frame_count}/{total_frames} | "
                f"Vehicles: {len(tracked)} | "
                f"Total IDs seen: {len(all_track_ids)}")
        cv2.putText(annotated, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Vehicle Tracking Test", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[Test] Quit by user.")
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[Test] Processed {frame_count} frames.")
    print(f"[Test] Unique vehicle IDs tracked: {len(all_track_ids)}")
    print(f"[Test] IDs seen: {sorted(all_track_ids)}")


# ── Run directly to test ───────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/tracker.py data/videos/traffic.mp4")
    else:
        test_tracker(sys.argv[1])