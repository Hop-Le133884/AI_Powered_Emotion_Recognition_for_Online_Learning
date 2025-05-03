# Multi-Person Engagement Analyzer

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.6%2B-green)

A real-time computer vision application that detects multiple people in a video stream and analyzes their engagement levels through facial emotion recognition.

## 🔍 Overview

This project combines YOLO (You Only Look Once) object detection with DeepFace emotion analysis to create a powerful engagement tracking system. The application:

1. Detects all people in the video frame using YOLO
2. Analyzes each person's facial emotions using DeepFace 
3. Determines engagement status based on emotion detection
4. Tracks individual "not engaged" time for each person
5. Provides visual feedback with bounding boxes and status information

Perfect for educational settings, user experience testing, or audience engagement analysis.

## ✨ Features

- **Multi-person tracking**: Detects and analyzes multiple people simultaneously
- **Facial emotion recognition**: Identifies 7 emotions (happy, sad, angry, surprise, fear, disgust, neutral)
- **Engagement status**: Classifies subjects as "Engaged" or "Not Engaged" in real-time
- **Engagement timing**: Tracks how long each person remains disengaged
- **Performance optimized**: Processes frames efficiently for real-time analysis
- **Visual feedback**: Color-coded bounding boxes and on-screen statistics

## 🛠️ Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/multi-person-engagement-analyzer.git
cd multi-person-engagement-analyzer
```

2. Install required dependencies
```bash
pip install -r requirements.txt
```

## 📋 Requirements

- Python 3.6+
- OpenCV
- PyTorch
- DeepFace
- Ultralytics YOLO
- NumPy
- CUDA-capable GPU (recommended for faster processing)

## 🚀 Usage

Run the main script:

```bash
cd src/
python yolo_emotion_detection.py #detecting person and face expression
```
``` bash (other) to explore
python emotion_reader.py
python yolo_person_detection.py
```

Controls:
- Press 'q' to quit the application

## 🔧 Configuration

You can adjust the following parameters in the script:

- `YOLO_MODEL`: Choose between different YOLO model sizes (nano, small, medium, etc.)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for person detection (0.0-1.0)
- `PROCESS_EVERY_N_FRAMES`: Skip frames for better performance
- `detector_backend`: Face detector backend for DeepFace ('opencv', 'ssd', 'mediapipe', etc.)
- `bbox_size_threshold`: Maximum face size relative to the frame (to filter false detections)

## 🧪 Engagement Logic

The system considers a person "Engaged" if any facial emotion is detected, including neutral expressions. A person is classified as "Not Engaged" when:

- No face is detected within their body region
- Face analysis encounters an error
- Face detection is filtered out (e.g., too large relative to the frame)

## 📊 Output

The application provides:

- Live video feed with bounding boxes
- Person identification
- Emotion labels with confidence percentages
- Engagement status per person
- Individual "not engaged" timers
- Summary

## 📁 Project Structure

```
.
├── models
│   └── yolov8n.pt
├── README.md
└── src
    ├── emotion_reader.py
    ├── mapper_emotion.py
    ├── __pycache__
    │   └── mapper_emotion.cpython-311.pyc
    ├── yolo_emotion_detection.py
    └── yolo_person_detection.py
```

### File Descriptions

- **emotion_reader.py**: DeepFace wrapper for emotion analysis
- **mapper_emotion.py**: Maps detected emotions to engagement states
- **yolo_emotion_detection.py**: Main application that combines YOLO and emotion analysis
- **yolo_person_detection.py**: Basic person detection using YOLO
- **models/yolov8n.pt**: Pre-trained YOLOv8 nano model