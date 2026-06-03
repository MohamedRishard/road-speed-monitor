# config/settings.py
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data", "videos")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
VIDEO_OUT   = os.path.join(OUTPUT_DIR, "videos")
FRAMES_OUT  = os.path.join(OUTPUT_DIR, "frames")
REPORTS_OUT = os.path.join(OUTPUT_DIR, "reports")

# ── Model ──────────────────────────────────────────────────────────────────
YOLO_MODEL  = "yolov8n.pt"
CONFIDENCE  = 0.4

# ── Tracking ───────────────────────────────────────────────────────────────
TRACKER     = "bytetrack.yaml"

# ── Speed Estimation ───────────────────────────────────────────────────────
SPEED_LIMIT_KMH  = 40 

# Reference lines — y coordinates from inspect_video.py
LINE_A_Y         = 266      # upper reference line
LINE_B_Y         = 571      # lower reference line

# Calibration: real-world distance between Line A and Line B
# Change this value based on your road knowledge
# Start with 40.0 and we can tune it later
LINE_DISTANCE_M  = 40.0     # meters between the two lines

# Calculated automatically — do not change
PIXELS_PER_METER = (LINE_B_Y - LINE_A_Y) / LINE_DISTANCE_M   # = 305/40 = ~7.6

FPS_OVERRIDE     = None

# ── Vehicle Classes (COCO dataset IDs) ────────────────────────────────────
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES  = [2, 3, 5, 7]

# ── Display ────────────────────────────────────────────────────────────────
NORMAL_COLOR     = (0, 255, 0)
OVERSPEED_COLOR  = (0, 0, 255)
FONT_SCALE       = 0.6
BOX_THICKNESS    = 2