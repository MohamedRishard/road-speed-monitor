# src/test_overspeed.py
"""
Combined test: Tracking + Speed Estimation + Overspeed Detection
Run this to see the full detection pipeline working together.
"""

import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracker         import VehicleTracker
from src.speed_estimator import SpeedEstimator, draw_reference_lines
from src.overspeed       import OverspeedDetector, draw_overspeed_overlay, draw_speed_limit_sign
from config.settings     import SPEED_LIMIT_KMH


def test_full_pipeline(video_path):
    print(f"\n[Pipeline Test] Video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Cannot open video.")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Pipeline Test] {width}x{height} @ {fps}fps | {total_frames} frames")
    print(f"[Pipeline Test] Speed limit: {SPEED_LIMIT_KMH} km/h")
    print("[Pipeline Test] Press 'q' to quit | SPACE to pause\n")

    tracker   = VehicleTracker()
    estimator = SpeedEstimator(fps)
    detector  = OverspeedDetector()

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 1. Track
        tracked = tracker.track(frame)

        # 2. Estimate speeds
        speeds = estimator.update(tracked, frame_count)

        # 3. Check overspeed
        detector.check(tracked, speeds, frame, frame_count)

        # 4. Draw everything
        annotated = frame.copy()
        annotated = draw_reference_lines(annotated)
        annotated = draw_overspeed_overlay(
            annotated, tracked, speeds,
            detector.alerted_ids, SPEED_LIMIT_KMH
        )
        annotated = draw_speed_limit_sign(annotated, SPEED_LIMIT_KMH)

        # Stats panel
        violations = detector.get_violation_count()
        info = (f"Frame:{frame_count}/{total_frames} | "
                f"Vehicles:{len(tracked)} | "
                f"Violations:{violations}")
        cv2.putText(annotated, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Overspeed Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

    # Final summary
    all_speeds   = estimator.get_all_speeds()
    violations   = detector.get_violations()

    print(f"\n{'='*50}")
    print(f"  FINAL RESULTS")
    print(f"{'='*50}")
    print(f"  Total vehicles tracked : {len(all_speeds)}")
    print(f"  Overspeed violations   : {len(violations)}")

    if all_speeds:
        vals = list(all_speeds.values())
        print(f"  Average speed          : {np.mean(vals):.1f} km/h")
        print(f"  Maximum speed          : {np.max(vals):.1f} km/h")

    if violations:
        print(f"\n  Violating vehicles:")
        for v in violations:
            print(f"    ID:{v['track_id']} | {v['class_name']} | "
                  f"{v['speed_kmh']} km/h | +{v['excess_kmh']} over limit")
    print(f"{'='*50}")
    print(f"  Evidence frames saved to: output/frames/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/test_overspeed.py data/videos/traffic.mp4")
    else:
        test_full_pipeline(sys.argv[1])