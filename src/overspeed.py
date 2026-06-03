# src/overspeed.py
"""
Overspeed Detection Module

Responsibilities:
- Compare vehicle speeds against speed limit
- Maintain a log of all violations
- Save evidence frames for violations
- Provide violation data for CSV reporting
"""

import cv2
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    SPEED_LIMIT_KMH, OVERSPEED_COLOR, NORMAL_COLOR,
    FONT_SCALE, BOX_THICKNESS, FRAMES_OUT
)


class OverspeedDetector:
    """
    Detects and logs overspeed violations.

    Maintains a violation log with:
    - Vehicle ID
    - Speed at time of violation
    - Timestamp
    - Frame number
    - Vehicle class
    """

    def __init__(self):
        self.speed_limit  = SPEED_LIMIT_KMH
        self.violations   = {}       # {track_id: violation_dict}
        self.alerted_ids  = set()    # IDs already logged (avoid duplicates)

        os.makedirs(FRAMES_OUT, exist_ok=True)
        print(f"[OverspeedDetector] Speed limit: {self.speed_limit} km/h")

    def check(self, tracked_vehicles, speeds, frame, frame_number):
        """
        Check all tracked vehicles for overspeed violations.

        Args:
            tracked_vehicles: list from VehicleTracker.track()
            speeds:           dict {track_id: speed_kmh}
            frame:            current BGR frame (for saving evidence)
            frame_number:     current frame index

        Returns:
            list of new violation dicts detected this frame
        """
        new_violations = []

        for vehicle in tracked_vehicles:
            tid   = vehicle['track_id']
            speed = speeds.get(tid)

            if speed is None:
                continue  # speed not yet confirmed for this vehicle

            if speed > self.speed_limit and tid not in self.alerted_ids:
                violation = {
                    'track_id':    tid,
                    'class_name':  vehicle['class_name'],
                    'speed_kmh':   speed,
                    'speed_limit': self.speed_limit,
                    'excess_kmh':  round(speed - self.speed_limit, 1),
                    'frame':       frame_number,
                    'timestamp':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'bbox':        vehicle['bbox']
                }

                self.violations[tid]  = violation
                self.alerted_ids.add(tid)
                new_violations.append(violation)

                # Save evidence frame
                self._save_evidence(frame, violation)

                print(f"  [OVERSPEED] Vehicle {tid} ({vehicle['class_name']}): "
                      f"{speed:.1f} km/h — "
                      f"{speed - self.speed_limit:.1f} km/h over limit!")

        return new_violations

    def _save_evidence(self, frame, violation):
        """Save annotated evidence frame for the violation."""
        evidence = frame.copy()
        tid       = violation['track_id']

        # Draw red box around violating vehicle
        x1, y1, x2, y2 = violation['bbox']
        cv2.rectangle(evidence, (x1, y1), (x2, y2), OVERSPEED_COLOR, 3)

        # Violation label
        label = (f"ID:{tid} | {violation['speed_kmh']} km/h | "
                 f"LIMIT: {violation['speed_limit']} km/h")
        cv2.putText(evidence, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, OVERSPEED_COLOR, 2)

        # Timestamp overlay
        cv2.putText(evidence, violation['timestamp'], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Big OVERSPEED warning banner
        h, w = evidence.shape[:2]
        cv2.rectangle(evidence, (0, h - 50), (w, h), (0, 0, 255), -1)
        cv2.putText(evidence, f"OVERSPEED VIOLATION — Vehicle {tid}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        filename   = f"violation_id{tid}_frame{violation['frame']}.jpg"
        save_path  = os.path.join(FRAMES_OUT, filename)
        cv2.imwrite(save_path, evidence)
        print(f"  [Evidence] Saved: {save_path}")

    def is_violator(self, track_id):
        """Check if a vehicle has been flagged as a violator."""
        return track_id in self.alerted_ids

    def get_violations(self):
        """Return all violations as a list of dicts."""
        return list(self.violations.values())

    def get_violation_count(self):
        """Return total number of unique violations."""
        return len(self.violations)


def draw_overspeed_overlay(frame, tracked_vehicles, speeds, violator_ids, speed_limit):
    """
    Draw speed-aware annotations on frame.

    - Red box + OVERSPEED label for violators
    - Green box + speed for normal vehicles
    - Speed displayed once confirmed

    Args:
        frame:           BGR image
        tracked_vehicles: list from tracker
        speeds:          dict {track_id: speed_kmh}
        violator_ids:    set of track_ids that violated
        speed_limit:     int km/h

    Returns:
        Annotated frame
    """
    for v in tracked_vehicles:
        tid        = v['track_id']
        x1, y1, x2, y2 = v['bbox']
        speed      = speeds.get(tid)
        is_over    = tid in violator_ids

        color = OVERSPEED_COLOR if is_over else NORMAL_COLOR

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Build label
        if speed is not None:
            speed_str = f"{speed:.1f}km/h"
            tag       = " !" if is_over else ""
            label     = f"ID:{tid} {v['class_name']} {speed_str}{tag}"
        else:
            label = f"ID:{tid} {v['class_name']}"

        # Label background
        (w, h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
        cv2.rectangle(frame, (x1, y1 - 22), (x1 + w + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 1)

        # OVERSPEED warning text below box
        if is_over:
            cv2.putText(frame, "OVERSPEED", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, OVERSPEED_COLOR, 2)

    return frame


def draw_speed_limit_sign(frame, speed_limit):
    """Draw a speed limit indicator in the corner of the frame."""
    h, w = frame.shape[:2]

    # Background circle
    cx, cy = w - 60, 60
    cv2.circle(frame, (cx, cy), 45, (255, 255, 255), -1)
    cv2.circle(frame, (cx, cy), 45, (0, 0, 200),     4)
    cv2.circle(frame, (cx, cy), 38, (0, 0, 200),     2)

    # Speed limit number
    text = str(speed_limit)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, text,
                (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # "km/h" label
    cv2.putText(frame, "km/h", (cx - 18, cy + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    return frame