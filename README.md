# 🚗 Road Vehicle Speed Monitoring & Overspeed Detection System

> A real-time computer vision pipeline that detects, tracks, and measures
> vehicle speeds from traffic footage — flagging overspeed violations with
> evidence frames, CSV logs, and statistics reports.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Features

- 🎯 **Vehicle Detection** — YOLOv8 detects cars, buses, trucks, motorcycles
- 🔢 **Persistent Tracking** — ByteTrack assigns stable IDs across frames
- ⚡ **Speed Estimation** — Two-line calibration method (mirrors real speed cameras)
- 🚨 **Overspeed Detection** — Configurable speed limit with red box alerts
- 🎬 **Annotated Video Output** — Full MP4 with overlays and dashboard
- 📊 **CSV Reports** — Speed log and violation log with timestamps
- 📈 **Statistics Charts** — Speed distribution and per-vehicle bar chart
- 🖼️ **Evidence Frames** — Saved JPG for each violation

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| YOLOv8 (Ultralytics) | Vehicle detection |
| ByteTrack | Multi-object tracking |
| OpenCV | Video I/O and annotation |
| NumPy | Speed calculations |
| Pandas | CSV reporting |
| Matplotlib | Statistics charts |

---

## 📁 Project Structure

road_speed_monitor/
├── src/
│   ├── detector.py          # YOLOv8 vehicle detection
│   ├── tracker.py           # ByteTrack integration
│   ├── speed_estimator.py   # Two-line speed calculation
│   ├── overspeed.py         # Violation detection + evidence saving
│   ├── annotator.py         # Output video writer
│   └── reporter.py          # CSV + charts generation
├── config/
│   └── settings.py          # All tunable parameters
├── data/videos/             # Input videos (not tracked by git)
├── output/
│   ├── videos/              # Annotated output videos
│   ├── frames/              # Violation evidence frames
│   └── reports/             # CSV logs + charts
├── tools/
│   └── inspect_video.py     # Video calibration helper
├── main.py                  # Pipeline entry point
└── requirements.txt

---

## ⚙️ Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/road-speed-monitor.git
cd road-speed-monitor

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
# Run full pipeline with live display
python main.py --video data/videos/traffic.mp4

# Run without display (faster, for long videos)
python main.py --video data/videos/traffic.mp4 --no-display

# Custom speed limit
python main.py --video data/videos/traffic.mp4 --limit 60

# Inspect video to calibrate reference lines
python tools/inspect_video.py data/videos/traffic.mp4
```

---

## 📊 Sample Output

==================================================
ROAD SPEED MONITORING — SESSION REPORT
Total vehicles   : 11
Average speed    : 38.7 km/h
Maximum speed    : 46.2 km/h
Violations       : 3
Violation rate   : 27.3%

---

## ⚙️ Configuration

Edit `config/settings.py` to tune the system:

```python
SPEED_LIMIT_KMH  = 40      # Speed limit threshold
LINE_A_Y         = 266     # Upper reference line (pixels)
LINE_B_Y         = 571     # Lower reference line (pixels)
LINE_DISTANCE_M  = 40.0    # Real-world distance between lines (meters)
CONFIDENCE       = 0.4     # Detection confidence threshold
```

---

## 📌 Limitations

- Speed accuracy depends on camera angle and calibration
- Works best with side-view or overhead traffic cameras
- Perspective distortion affects accuracy at extreme angles

---

