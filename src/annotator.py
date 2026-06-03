# src/annotator.py
"""
Video Annotation Module

Handles writing the final annotated output video with:
- Bounding boxes and speed labels
- Reference lines
- Speed limit sign
- Stats dashboard overlay
- Overspeed warnings
"""

import cv2
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    VIDEO_OUT, SPEED_LIMIT_KMH,
    NORMAL_COLOR, OVERSPEED_COLOR,
    FONT_SCALE, BOX_THICKNESS,
    LINE_A_Y, LINE_B_Y
)


class VideoAnnotator:
    """
    Writes annotated frames to an output video file.
    """

    def __init__(self, source_path, fps, width, height):
        os.makedirs(VIDEO_OUT, exist_ok=True)

        base      = os.path.splitext(os.path.basename(source_path))[0]
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path  = os.path.join(VIDEO_OUT, f"{base}_annotated_{timestamp}.mp4")

        fourcc        = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        self.out_path = out_path
        self.width    = width
        self.height   = height

        print(f"[Annotator] Output video: {out_path}")

    def write(self, frame, tracked_vehicles, speeds,
              violator_ids, frame_number, total_vehicles, violations_count):
        """
        Annotate and write a single frame.
        """
        annotated = frame.copy()

        # Reference lines
        annotated = self._draw_lines(annotated)

        # Vehicle boxes and labels
        annotated = self._draw_vehicles(annotated, tracked_vehicles,
                                        speeds, violator_ids)

        # Speed limit sign
        annotated = self._draw_speed_sign(annotated)

        # Dashboard
        annotated = self._draw_dashboard(
            annotated, frame_number, len(tracked_vehicles),
            total_vehicles, violations_count, speeds
        )

        self.writer.write(annotated)

    def _draw_lines(self, frame):
        cv2.line(frame, (0, LINE_A_Y), (self.width, LINE_A_Y),
                 (255, 100, 0), 2)
        cv2.putText(frame, "Line A", (10, LINE_A_Y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        cv2.line(frame, (0, LINE_B_Y), (self.width, LINE_B_Y),
                 (0, 165, 255), 2)
        cv2.putText(frame, "Line B", (10, LINE_B_Y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        return frame

    def _draw_vehicles(self, frame, tracked, speeds, violator_ids):
        for v in tracked:
            tid             = v['track_id']
            x1, y1, x2, y2 = v['bbox']
            speed           = speeds.get(tid)
            is_over         = tid in violator_ids
            color           = OVERSPEED_COLOR if is_over else NORMAL_COLOR

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            if speed:
                label = f"ID:{tid} {speed:.1f}km/h"
                if is_over:
                    label += " !"
            else:
                label = f"ID:{tid} {v['class_name']}"

            (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
            cv2.rectangle(frame,
                          (x1, y1 - 22), (x1 + w + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SCALE, (0, 0, 0), 1)

            if is_over:
                cv2.putText(frame, "OVERSPEED",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, OVERSPEED_COLOR, 2)
        return frame

    def _draw_speed_sign(self, frame):
        cx, cy = self.width - 60, 60
        cv2.circle(frame, (cx, cy), 45, (255, 255, 255), -1)
        cv2.circle(frame, (cx, cy), 45, (0, 0, 200), 4)
        cv2.circle(frame, (cx, cy), 38, (0, 0, 200), 2)
        text = str(SPEED_LIMIT_KMH)
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(frame, text,
                    (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(frame, "km/h", (cx - 18, cy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        return frame

    def _draw_dashboard(self, frame, frame_num, active,
                        total, violations, speeds):
        # Semi-transparent dark bar at bottom
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (0, self.height - 40),
                      (self.width, self.height),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        avg_spd = (np.mean(list(speeds.values()))
                   if speeds else 0)

        dash = (f"Frame:{frame_num} | "
                f"Active:{active} | "
                f"Total:{total} | "
                f"Avg:{avg_spd:.1f}km/h | "
                f"Violations:{violations}")
        cv2.putText(frame, dash,
                    (10, self.height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
        return frame

    def release(self):
        self.writer.release()
        print(f"[Annotator] Video saved: {self.out_path}")
        return self.out_path