# 🚗 Road Vehicle Speed Monitoring & Overspeed Detection System

A real-time computer vision system that detects, tracks, and measures vehicle
speeds from traffic video footage using YOLOv8 and ByteTrack.

## Features
- Vehicle detection (cars, buses, trucks, motorcycles) via YOLOv8
- Multi-object tracking with unique IDs via ByteTrack
- Real-time speed estimation
- Overspeed violation detection and highlighting
- CSV violation logs with timestamps
- Annotated output video generation
- Summary statistics (avg speed, max speed, violation count)

## Tech Stack
Python · OpenCV · YOLOv8 · ByteTrack · NumPy · Pandas · Matplotlib

## Setup
```bash
python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
python main.py --video data/videos/traffic.mp4
```

## Output
Annotated video, CSV violation log, and statistics chart saved to `/output/`.