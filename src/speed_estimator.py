# src/speed_estimator.py
"""
Speed Estimation Module

How it works:
1. Track each vehicle's center point every frame
2. When vehicle crosses Line A — record frame number
3. When vehicle crosses Line B — record frame number
4. Speed = LINE_DISTANCE_M / time_between_crossings

Why two lines?
- More accurate than frame-to-frame pixel displacement
- Eliminates noise from small detection jitter
- Mirrors real-world speed cameras (two sensors on road)
"""

import sys
import os
import cv2
import numpy as np
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    LINE_A_Y, LINE_B_Y, LINE_DISTANCE_M,
    PIXELS_PER_METER, FPS_OVERRIDE
)


class SpeedEstimator:
    """
    Estimates vehicle speed using two reference lines.

    Tracks when each vehicle crosses Line A and Line B,
    then calculates speed from time elapsed between crossings.
    """

    def __init__(self, fps):
        self.fps           = FPS_OVERRIDE if FPS_OVERRIDE else fps
        self.line_a_y      = LINE_A_Y
        self.line_b_y      = LINE_B_Y
        self.line_dist_m   = LINE_DISTANCE_M
        self.px_per_meter  = PIXELS_PER_METER

        # Per-vehicle state tracking
        # {track_id: {'crossed_a': frame_num, 'crossed_b': frame_num}}
        self.crossing_data  = defaultdict(dict)

        # Final confirmed speeds {track_id: speed_kmh}
        self.confirmed_speeds = {}

        # Position history for smooth display {track_id: [centers]}
        self.position_history = defaultdict(list)

        print(f"[SpeedEstimator] FPS: {self.fps}")
        print(f"[SpeedEstimator] Line A: y={self.line_a_y}px | "
              f"Line B: y={self.line_b_y}px")
        print(f"[SpeedEstimator] Real distance: {self.line_dist_m}m | "
              f"Scale: {self.px_per_meter:.2f} px/m")

    def update(self, tracked_vehicles, frame_number):
        """
        Process tracked vehicles for current frame.

        Args:
            tracked_vehicles: list of dicts from VehicleTracker.track()
            frame_number: current frame index

        Returns:
            dict of {track_id: speed_kmh} for all vehicles with known speed
        """
        for vehicle in tracked_vehicles:
            tid = vehicle['track_id']
            cx, cy = vehicle['center']

            # Store position history (keep last 30 positions)
            self.position_history[tid].append((cx, cy, frame_number))
            if len(self.position_history[tid]) > 30:
                self.position_history[tid].pop(0)

            # Check Line A crossing (vehicle enters measurement zone)
            if 'crossed_a' not in self.crossing_data[tid]:
                if abs(cy - self.line_a_y) < 12:   # within 12px of line
                    self.crossing_data[tid]['crossed_a'] = frame_number
                    self.crossing_data[tid]['pos_a']     = cy

            # Check Line B crossing (vehicle exits measurement zone)
            elif 'crossed_b' not in self.crossing_data[tid]:
                if abs(cy - self.line_b_y) < 12:   # within 12px of line
                    self.crossing_data[tid]['crossed_b'] = frame_number
                    self.crossing_data[tid]['pos_b']     = cy

                    # Calculate speed now that we have both crossings
                    speed = self._calculate_speed(tid)
                    if speed is not None:
                        self.confirmed_speeds[tid] = speed

        return self.confirmed_speeds

    def _calculate_speed(self, track_id):
        """
        Calculate speed for a vehicle that crossed both lines.

        Returns:
            speed in km/h, or None if calculation invalid
        """
        data = self.crossing_data[track_id]

        if 'crossed_a' not in data or 'crossed_b' not in data:
            return None

        frames_elapsed = data['crossed_b'] - data['crossed_a']

        if frames_elapsed <= 0:
            return None

        time_seconds = frames_elapsed / self.fps
        speed_ms     = self.line_dist_m / time_seconds
        speed_kmh    = speed_ms * 3.6

        # Sanity check — ignore impossible speeds
        if speed_kmh < 1 or speed_kmh > 300:
            return None

        print(f"  [Speed] Vehicle {track_id}: "
              f"{frames_elapsed} frames | "
              f"{time_seconds:.2f}s | "
              f"{speed_kmh:.1f} km/h")

        return round(speed_kmh, 1)

    def get_speed(self, track_id):
        """Get last known speed for a vehicle."""
        return self.confirmed_speeds.get(track_id, None)

    def get_all_speeds(self):
        """Return all confirmed speeds."""
        return dict(self.confirmed_speeds)


def draw_reference_lines(frame):
    """
    Draw the two reference lines on the frame.
    Call this every frame so lines are always visible.
    """
    h, w = frame.shape[:2]

    # Line A — blue
    cv2.line(frame, (0, LINE_A_Y), (w, LINE_A_Y), (255, 100, 0), 2)
    cv2.putText(frame, "Line A", (10, LINE_A_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    # Line B — orange
    cv2.line(frame, (0, LINE_B_Y), (w, LINE_B_Y), (0, 165, 255), 2)
    cv2.putText(frame, "Line B", (10, LINE_B_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    return frame


def test_speed_estimator(video_path):
    """
    Test speed estimation — watch speeds appear as vehicles cross lines.
    """
    from src.tracker import VehicleTracker, draw_tracked_vehicles

    print(f"\n[Test] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker   = VehicleTracker()
    estimator = SpeedEstimator(fps)

    print(f"[Test] Press 'q' to quit | SPACE to pause")
    print(f"[Test] Speeds will appear once vehicles cross BOTH lines\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Track vehicles
        tracked = tracker.track(frame)

        # Update speed estimates
        speeds = estimator.update(tracked, frame_count)

        # Draw everything
        annotated = frame.copy()
        annotated = draw_reference_lines(annotated)
        annotated = draw_tracked_vehicles(annotated, tracked, speeds)

        # Stats overlay
        confirmed = len(speeds)
        info = (f"Frame: {frame_count}/{total_frames} | "
                f"Vehicles: {len(tracked)} | "
                f"Speeds confirmed: {confirmed}")
        cv2.putText(annotated, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Speed Estimation Test", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

    # Print final speed summary
    all_speeds = estimator.get_all_speeds()
    print(f"\n[Results] Speed measurements captured: {len(all_speeds)}")
    if all_speeds:
        speeds_list = list(all_speeds.values())
        print(f"[Results] Speeds (km/h): {all_speeds}")
        print(f"[Results] Average: {np.mean(speeds_list):.1f} km/h")
        print(f"[Results] Maximum: {np.max(speeds_list):.1f} km/h")
        print(f"[Results] Minimum: {np.min(speeds_list):.1f} km/h")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/speed_estimator.py data/videos/traffic.mp4")
    else:
        test_speed_estimator(sys.argv[1])