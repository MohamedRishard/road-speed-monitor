# main.py
"""
Road Vehicle Speed Monitoring & Overspeed Detection System
Main pipeline entry point.

Usage:
    python main.py --video data/videos/traffic.mp4
    python main.py --video data/videos/traffic.mp4 --limit 50
    python main.py --video data/videos/traffic.mp4 --no-display
"""

import cv2
import sys
import os
import argparse
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.tracker         import VehicleTracker
from src.speed_estimator import SpeedEstimator, draw_reference_lines
from src.overspeed       import (OverspeedDetector, draw_overspeed_overlay,
                                  draw_speed_limit_sign)
from src.annotator       import VideoAnnotator
from src.reporter        import Reporter
from config.settings     import SPEED_LIMIT_KMH


def parse_args():
    parser = argparse.ArgumentParser(
        description='Road Vehicle Speed Monitoring System')
    parser.add_argument('--video',      required=True,
                        help='Path to input video file')
    parser.add_argument('--limit',      type=int, default=SPEED_LIMIT_KMH,
                        help=f'Speed limit in km/h (default: {SPEED_LIMIT_KMH})')
    parser.add_argument('--no-display', action='store_true',
                        help='Run without showing video window (faster)')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("   ROAD VEHICLE SPEED MONITORING SYSTEM")
    print("=" * 55)
    print(f"   Video      : {args.video}")
    print(f"   Speed limit: {args.limit} km/h")
    print(f"   Display    : {'OFF' if args.no_display else 'ON'}")
    print("=" * 55 + "\n")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Main] Video: {width}x{height} @ {fps}fps | {total_frames} frames")

    # Initialize all modules
    tracker    = VehicleTracker()
    estimator  = SpeedEstimator(fps)
    detector   = OverspeedDetector()
    annotator  = VideoAnnotator(args.video, fps, width, height)
    reporter   = Reporter(os.path.basename(args.video))

    # Speed tracking — avoid duplicate reporter entries
    reported_speeds = set()
    total_vehicles  = set()
    frame_count     = 0

    print(f"\n[Main] Processing video... Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"\n[Main] Video processing complete.")
            break

        frame_count += 1

        # ── 1. Track vehicles ─────────────────────────────────────
        tracked = tracker.track(frame)
        for v in tracked:
            total_vehicles.add(v['track_id'])

        # ── 2. Estimate speeds ────────────────────────────────────
        speeds = estimator.update(tracked, frame_count)

        # ── 3. Log new speed confirmations to reporter ────────────
        for tid, spd in speeds.items():
            if tid not in reported_speeds:
                reported_speeds.add(tid)
                vehicle_class = next(
                    (v['class_name'] for v in tracked
                     if v['track_id'] == tid), 'vehicle'
                )
                reporter.add_speed_record(tid, vehicle_class,
                                          spd, frame_count)

        # ── 4. Check overspeed ────────────────────────────────────
        new_violations = detector.check(
            tracked, speeds, frame, frame_count)
        for v in new_violations:
            reporter.add_violation(v)

        # ── 5. Write annotated frame to output video ──────────────
        annotator.write(
            frame, tracked, speeds,
            detector.alerted_ids, frame_count,
            len(total_vehicles),
            detector.get_violation_count()
        )

        # ── 6. Live display (optional) ────────────────────────────
        if not args.no_display:
            display = frame.copy()
            display = draw_reference_lines(display)
            display = draw_overspeed_overlay(
                display, tracked, speeds,
                detector.alerted_ids, args.limit)
            display = draw_speed_limit_sign(display, args.limit)

            cv2.putText(display,
                        f"Frame:{frame_count}/{total_frames} | "
                        f"Vehicles:{len(tracked)} | "
                        f"Violations:{detector.get_violation_count()}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)

            cv2.imshow("Road Speed Monitor", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[Main] Stopped by user.")
                break

        # Progress every 100 frames
        if frame_count % 100 == 0:
            pct = frame_count / total_frames * 100
            print(f"  Progress: {frame_count}/{total_frames} "
                  f"({pct:.0f}%) | "
                  f"Tracked: {len(total_vehicles)} vehicles | "
                  f"Violations: {detector.get_violation_count()}")

    # ── Cleanup ───────────────────────────────────────────────────
    cap.release()
    annotator.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # ── Generate all reports ──────────────────────────────────────
    reporter.generate_all()

    print("\n[Main] Pipeline finished successfully!")
    print(f"[Main] Output video  : output/videos/")
    print(f"[Main] Reports       : output/reports/")
    print(f"[Main] Evidence      : output/frames/")


if __name__ == "__main__":
    main()