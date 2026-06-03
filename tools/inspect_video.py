# tools/inspect_video.py
"""
Helper tool to inspect video and find good reference line positions.
- Shows frame with crosshair on mouse position
- Click to print pixel coordinates
- Press 's' to save current frame as image
- Press 'q' to quit
"""

import cv2
import sys
import os

mouse_x, mouse_y = 0, 0

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"  Clicked: x={x}, y={y}")

def inspect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nVideo: {width}x{height} @ {fps}fps")
    print("Controls: SPACE=pause/resume | S=save frame | Q=quit")
    print("Click anywhere to print coordinates\n")

    cv2.namedWindow("Inspect Video")
    cv2.setMouseCallback("Inspect Video", mouse_callback)

    paused      = False
    frame_count = 0
    frame       = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Video ended.")
                break
            frame_count += 1

        if frame is None:
            continue

        display = frame.copy()

        # Draw crosshair at mouse position
        cv2.line(display, (mouse_x, 0), (mouse_x, height), (0, 255, 255), 1)
        cv2.line(display, (0, mouse_y), (width, mouse_y),  (0, 255, 255), 1)

        # Show coordinates
        coord_text = f"x={mouse_x}, y={mouse_y}"
        cv2.putText(display, coord_text, (mouse_x + 10, mouse_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Frame info
        info = f"Frame: {frame_count} | {width}x{height} @ {fps}fps"
        cv2.putText(display, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if paused:
            cv2.putText(display, "PAUSED", (width//2 - 50, height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow("Inspect Video", display)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"  {'Paused' if paused else 'Resumed'} at frame {frame_count}")
        elif key == ord('s') and frame is not None:
            save_path = f"output/frames/inspect_frame_{frame_count}.jpg"
            os.makedirs("output/frames", exist_ok=True)
            cv2.imwrite(save_path, frame)
            print(f"  Saved: {save_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/inspect_video.py data/videos/traffic.mp4")
    else:
        inspect_video(sys.argv[1])