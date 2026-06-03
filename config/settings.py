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
SPEED_LIMIT_KMH  = 60
PIXELS_PER_METER = 8.0
FPS_OVERRIDE     = None

# ── Vehicle Classes (COCO dataset IDs) ────────────────────────────────────
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES  = [2, 3, 5, 7]

# ── Display ────────────────────────────────────────────────────────────────
NORMAL_COLOR     = (0, 255, 0)
OVERSPEED_COLOR  = (0, 0, 255)
FONT_SCALE       = 0.6
BOX_THICKNESS    = 2