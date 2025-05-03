# 🔍 Overview
Overview: A real-time computer vision application that detects multiple people in a video stream and analyzes their engagement levels through facial emotion recognition.


## This project has two version:

## Webcam: Multi-Person Engagement Analyzer

![engaged_neutral_emotion](https://github.com/user-attachments/assets/e38ff29b-7f76-4928-b758-409aaf0182bb)
![not_engaged](https://github.com/user-attachments/assets/32a80cfd-3d2c-4778-abf4-6f1ca762809b)

## screen capture: Meeting Engagement Analyzer

This application captures your primary monitor screen to analyze engagement levels of multiple participants in a Zoom meeting or other video conference software. It uses computer vision to detect people on screen and analyzes their facial emotions to determine engagement status in real-time.

This project combines YOLO (You Only Look Once) object detection with DeepFace emotion analysis to create a powerful engagement tracking system. The application:



## Multi-Person Engagement Analyzer

1. Detects all people in the video frame using YOLO
2. Analyzes each person's facial emotions using DeepFace 
3. Determines engagement status based on emotion detection
4. Tracks individual "not engaged" time for each person
5. Provides visual feedback with bounding boxes and status information

## Zoom Meeting Engagement Analyzer

Screen capture: Automatically captures your primary monitor
Multi-person tracking: Detects all visible participants in the meeting
Facial emotion analysis: Identifies 7 emotions per participant
Engagement metrics: Calculates overall meeting engagement percentage
Individual tracking: Monitors not-engaged time for each participant
Session logging: Records engagement statistics over time at regular intervals
Performance optimized: Scale factor and frame skipping for efficiency


Perfect for educational settings, user experience testing, or audience engagement analysis.

## Configuration
You can adjust these parameters in the script to match your setup:
python# Screen Capture Configuration
monitor_number = 1                 # Primary monitor (1-based index)
screen_capture_width = 1920        # Adjust to your screen resolution
screen_capture_height = 1080       # Adjust to your screen resolution
screen_scale_factor = 0.75         # Scale factor for processing (1.0 = full resolution)

# YOLO Configuration
YOLO_MODEL = 'yolov8n.pt'          # YOLOv8 model size (nano is fastest)
CONFIDENCE_THRESHOLD = 0.5         # Person detection confidence threshold
PROCESS_EVERY_N_FRAMES = 3         # Process only every Nth frame for better performance

# DeepFace Configuration
detector_backend = 'opencv'        # Face detector backend ('opencv', 'ssd', 'mediapipe')
bbox_size_threshold = 0.54         # Filter out faces that are too large
## ✨ Features

- **Multi-person tracking**: Detects and analyzes multiple people simultaneously
- **Facial emotion recognition**: Identifies 7 emotions (happy, sad, angry, surprise, fear, disgust, neutral)
- **Engagement status**: Classifies subjects as "Engaged" or "Not Engaged" in real-time
- **Engagement timing**: Tracks how long each person remains disengaged
- **Performance optimized**: Processes frames efficiently for real-time analysis
- **Visual feedback**: Color-coded bounding boxes and on-screen statistics

## Limitations

Works best with clearly visible faces in good lighting
Performance depends on your hardware capabilities
"Not engaged" determination is based solely on face detection and emotion analysis
May not work with virtual backgrounds that interfere with person detection

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
python zoom_engagement_analyzer.py 
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

## Tips for Best Results

Make sure the Zoom gallery view is visible on your primary monitor
Ensure good lighting for all participants
Adjust the screen_scale_factor for balance between performance and accuracy
For better performance, consider using a smaller YOLO model or increasing PROCESS_EVERY_N_FRAMES

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
