import cv2
import torch
from ultralytics import YOLO
import numpy as np
import time
from deepface import DeepFace
import mss
import mss.tools

# =========== Engagement Mapping Function ===========
def map_emotion_to_engagement(emotion_label):
    """
    Maps the detected emotion label to an engagement state.
    
    Args:
        emotion_label (str or None): The dominant emotion detected by DeepFace,
                                     or None/Error if no face/emotion found.
    
    Returns:
        tuple: A tuple containing:
            - str: The engagement state ('Engaged' or 'Not Engaged').
            - str: A reason string providing context (e.g., emotion detected or why not engaged).
    """
    valid_emotions = {'sad', 'angry', 'surprise', 'fear', 'happy', 'neutral', 'disgust'}
    if emotion_label and emotion_label.lower() in valid_emotions:
        return 'Engaged', f"({emotion_label} Detected)"
    elif emotion_label == "Error":
        return 'Not Engaged', "(Analysis Error)"
    else:
        return 'Not Engaged', "(No Face Detected)"

# ============= Configuration =============
# YOLO Configuration
YOLO_MODEL = 'models/yolov8n.pt'          # YOLOv8 Nano model (COCO dataset)
CONFIDENCE_THRESHOLD = 0.5         # Minimum detection confidence
PROCESS_EVERY_N_FRAMES = 3         # Process every 3rd frame for performance

# DeepFace Configuration
detector_backend = 'opencv'        # Fast detector for DeepFace
bbox_size_threshold = 0.54         # Ignore detections wider/taller than this fraction of frame

# Screen Capture Configuration
monitor_number = 1                 # Primary monitor (1-based index)
screen_capture_width = 920        # Adjust to your screen resolution
screen_capture_height = 480       # Adjust to your screen resolution
screen_scale_factor = 0.5         # Scale factor for processing (1.0 = full resolution)

# Initialize YOLO
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
yolo_model = YOLO(YOLO_MODEL).to(device)

# Initialize screen capture
with mss.mss() as sct:
    # Get monitor information
    monitors = sct.monitors
    if len(monitors) < monitor_number:
        print(f"Error: Monitor {monitor_number} not found. Using primary monitor.")
        monitor = monitors[0]
    else:
        monitor = monitors[monitor_number]  # Primary monitor
    
    print(f"Capturing from monitor: {monitor}")
    
    # Define capture area (you can modify this to capture just the Zoom window area)
    monitor_area = {
        "left": monitor["left"],
        "top": monitor["top"],
        "width": monitor["width"],
        "height": monitor["height"]
    }

    # Scale down for performance if needed
    process_width = int(monitor_area["width"] * screen_scale_factor)
    process_height = int(monitor_area["height"] * screen_scale_factor)

print("Screen capture initialized.")
print(f"Using '{detector_backend}' backend for face analysis.")
print("Engagement Rule: Engaged if ANY emotion (incl. neutral) is detected, Not Engaged otherwise.")
print("Processing multiple people from screen capture.")
print("Press 'q' to quit.")

# Initialize variables
frame_count = 0
last_results = []
start_time = time.monotonic()
prev_frame_time = start_time
person_engagement_timers = {}  # Dictionary to track not engaged time for each person
session_start_time = time.time()  # For logging engagement over time

# For tracking engagement stats over time (every minute)
engagement_log = []
last_log_time = session_start_time

# ============= Main Processing Loop =============
with mss.mss() as sct:
    while True:
        # Capture screen
        screenshot = sct.grab(monitor_area)
        frame = np.array(screenshot)
        
        # Convert from BGRA to BGR (remove alpha channel)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Resize for faster processing
        frame = cv2.resize(frame, (process_width, process_height))
        
        frame_display = frame.copy()
        current_results = []
        
        # Time Calculation
        new_frame_time = time.monotonic()
        delta_time = max(0.001, new_frame_time - prev_frame_time)
        fps = 1.0 / delta_time
        
        # Get frame dimensions for face detection size check
        frame_h, frame_w = frame.shape[:2]
        
        # Process every Nth frame with YOLO
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # YOLO Person Detection
            results = yolo_model.predict(
                frame,
                classes=[0],  # Person class
                conf=CONFIDENCE_THRESHOLD,
                device=device,
                verbose=False
            )
            
            # Process each detected person
            for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box
                person_id = f"person_{i}"  # Unique ID for each detected person
                
                # Initialize timer for new person
                if person_id not in person_engagement_timers:
                    person_engagement_timers[person_id] = 0.0
                
                # Extract person ROI for face analysis
                person_roi = frame[y1:y2, x1:x2]
                
                # Skip if ROI is empty or too small
                if person_roi.size == 0 or person_roi.shape[0] < 20 or person_roi.shape[1] < 20:
                    current_results.append({
                        'person_id': person_id,
                        'body_box': (x1, y1, x2, y2),
                        'engagement_state': 'Not Engaged',
                        'reason': "(ROI too small)",
                        'face_box': None,
                        'emotion': None
                    })
                    continue
                    
                # Initialize face analysis variables
                dominant_emotion = None
                face_region = None
                confidence_score = 0.0
                is_valid_detection = False
                analysis_error = False
                
                try:
                    # Analyze face and emotion in the person ROI
                    analysis_results = DeepFace.analyze(
                        img_path=person_roi,
                        actions=['emotion'],
                        detector_backend=detector_backend,
                        enforce_detection=False,
                        silent=True
                    )
                    
                    # Process face analysis results
                    if isinstance(analysis_results, list) and len(analysis_results) > 0:
                        first_face_result = analysis_results[0]
                        temp_face_region = first_face_result['region']
                        w, h = temp_face_region['w'], temp_face_region['h']
                        
                        # Bounding box sanity check (face shouldn't be too large relative to person ROI)
                        roi_w, roi_h = person_roi.shape[1], person_roi.shape[0]
                        if w < bbox_size_threshold * roi_w and h < bbox_size_threshold * roi_h and w > 0 and h > 0:
                            is_valid_detection = True
                            face_region = {
                                'x': temp_face_region['x'] + x1,  # Adjust to original frame coordinates
                                'y': temp_face_region['y'] + y1,
                                'w': temp_face_region['w'],
                                'h': temp_face_region['h']
                            }
                            dominant_emotion = first_face_result['dominant_emotion']
                            all_emotion_scores = first_face_result['emotion']
                            confidence_score = all_emotion_scores.get(dominant_emotion, 0) / 100.0
                
                except Exception as e:
                    # print(f"Error analyzing person {person_id}: {e}")
                    dominant_emotion = "Error"
                    analysis_error = True
                    is_valid_detection = False
                
                # Determine engagement state
                engagement_state, reason = map_emotion_to_engagement(dominant_emotion)
                
                # Update engagement timer
                if engagement_state == 'Not Engaged':
                    person_engagement_timers[person_id] += delta_time
                
                # Store results
                current_results.append({
                    'person_id': person_id,
                    'body_box': (x1, y1, x2, y2),
                    'face_box': face_region,
                    'engagement_state': engagement_state,
                    'reason': reason,
                    'emotion': dominant_emotion,
                    'confidence': confidence_score,
                    'is_valid_detection': is_valid_detection
                })
            
            # Update last results
            last_results = current_results
        
        # Display FPS
        cv2.putText(frame_display, f"FPS: {fps:.1f}", (5, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Draw results for each person
        engaged_count = 0
        not_engaged_count = 0
        
        for i, res in enumerate(last_results):
            # Extract info
            x1, y1, x2, y2 = res['body_box']
            person_id = res['person_id']
            engagement_state = res['engagement_state']
            reason = res['reason']
            
            # Count engagement stats
            if engagement_state == 'Engaged':
                engaged_count += 1
            else:
                not_engaged_count += 1
            
            # Colors: Blue for body box, Green for engaged, Red for not engaged
            body_color = (255, 0, 0)  # Blue for body
            state_color = (0, 255, 0) if engagement_state == 'Engaged' else (0, 0, 255)
            
            # Draw body rectangle
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), body_color, 2)
            
            # Draw face rectangle if available and valid
            if res.get('is_valid_detection', False) and res.get('face_box'):
                face_region = res['face_box']
                fx, fy, fw, fh = face_region['x'], face_region['y'], face_region['w'], face_region['h']
                cv2.rectangle(frame_display, (fx, fy), (fx+fw, fy+fh), state_color, 2)
                
                # Show emotion if detected
                if res.get('emotion'):
                    emotion_text = f"{res['emotion']}: {res.get('confidence', 0)*100:.1f}%"
                    cv2.putText(frame_display, emotion_text, (fx, fy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
            
            # Display person ID and engagement state
            cv2.putText(frame_display, f"Person {i+1}: {engagement_state} {reason}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
            
            # Show not engaged time
            not_engaged_time = person_engagement_timers.get(person_id, 0.0)
            cv2.putText(frame_display, f"Not Engaged: {not_engaged_time:.1f}s", 
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Calculate overall meeting metrics
        total_people = len(last_results)
        engagement_percentage = 0 if total_people == 0 else (engaged_count / total_people) * 100
        
        # Display overall engagement stats
        cv2.putText(frame_display, f"Meeting Engagement: {engagement_percentage:.1f}% ({engaged_count}/{total_people})", 
                    (10, frame_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        session_duration = time.time() - session_start_time
        hours, remainder = divmod(int(session_duration), 3600)
        minutes, seconds = divmod(remainder, 60)
        cv2.putText(frame_display, f"Session Time: {hours:02d}:{minutes:02d}:{seconds:02d}", 
                    (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Log engagement stats every minute
        current_time = time.time()
        if current_time - last_log_time >= 60:  # Log every 60 seconds
            engagement_log.append({
                'timestamp': current_time,
                'engaged_count': engaged_count,
                'not_engaged_count': not_engaged_count,
                'total_people': total_people,
                'engagement_percentage': engagement_percentage
            })
            last_log_time = current_time
        
        # Display the result
        cv2.imshow('Zoom Meeting Engagement Analyzer', frame_display)
        
        # Update frame time and count
        prev_frame_time = new_frame_time
        frame_count += 1
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
print("Exiting...")
cv2.destroyAllWindows()
print("Resources released.")

# Display final engagement statistics
if person_engagement_timers:
    print("\n===== Engagement Statistics =====")
    for person_id, time_not_engaged in person_engagement_timers.items():
        print(f"{person_id}: Total Not Engaged time = {time_not_engaged:.2f} seconds")

# Display session engagement log
if engagement_log:
    print("\n===== Session Engagement Log =====")
    print("Time\t\tEngaged\tNot Engaged\tTotal\tEngagement %")
    for entry in engagement_log:
        log_time = time.strftime("%H:%M:%S", time.localtime(entry['timestamp']))
        print(f"{log_time}\t{entry['engaged_count']}\t{entry['not_engaged_count']}\t\t{entry['total_people']}\t{entry['engagement_percentage']:.1f}%")