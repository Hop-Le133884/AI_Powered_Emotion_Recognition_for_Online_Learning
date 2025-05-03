# ----- DeepFace Real-time Analyzer V4 (Simplified Engagement + Not Engaged Timer) -----
import cv2
from deepface import DeepFace
import time
from mapper_emotion import map_emotion_to_engagement


# --- Configuration ---
# Use a fast detector. 'ssd' or 'mediapipe' are good choices.
detector_backend = 'opencv'
# --- Bounding Box Filter Threshold ---
# Ignore detections wider or taller than this fraction of the frame
bbox_size_threshold = 0.54 # e.g., 90% - Adjust if needed
# --- End Configuration ---

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
print(f"Webcam opened. Using '{detector_backend}' backend. Applying BBox size filter (threshold: {bbox_size_threshold*100:.0f}%).")
print("Engagement Rule: Engaged if ANY emotion (incl. neutral) is detected, Not Engaged otherwise.")
print("Starting real-time analysis loop...")

# --- Initialize Time Tracking ---
# Use time.monotonic() for measuring time intervals reliably
start_time = time.monotonic()
prev_frame_time = start_time
total_not_engaged_time = 0.0  # Accumulator for Not Engaged time

# --- Real-time Processing Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
    
    # --- Time Calculation for Current Frame ---
    new_frame_time = time.monotonic()
    # Calculate time elapsed since the previous frame
    delta_time = new_frame_time - prev_frame_time
    # Prevent division by zero or negative time if clock adjustments happen
    delta_time = max(0.001, delta_time)
    fps = 1.0 / delta_time


    frame_h, frame_w = frame.shape[:2] # Get frame dimensions for size check

    # --- Face Detection and Emotion Analysis ---
    dominant_emotion = None # Reset emotion for the frame
    face_region = None      # Reset face region
    confidence_score = 0.0  # Reset confidence (though not used for mapping now)
    is_valid_detection = False # Flag for valid detection after filtering
    analysis_error = False     # Flag for analysis errors

    try:
        # Use DeepFace to find faces and analyze emotion in one go
        analysis_results = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            detector_backend=detector_backend,
            enforce_detection=False, # Don't error if no face found
            silent=True # Suppress DeepFace console logs
        )

        # DeepFace returns a list of dictionaries, one per detected face
        if isinstance(analysis_results, list) and len(analysis_results) > 0:
            # Process only the first detected face
            first_face_result = analysis_results[0]
            temp_face_region = first_face_result['region'] # {'x': ..., 'y': ..., 'w': ..., 'h': ...}
            w, h = temp_face_region['w'], temp_face_region['h']

            # --- Bounding Box Sanity Check ---
            # Check if the detected face bounding box is reasonably sized
            if w < bbox_size_threshold * frame_w and h < bbox_size_threshold * frame_h and w > 0 and h > 0 :
                # If box size is reasonable, consider it a valid detection
                is_valid_detection = True
                face_region = temp_face_region # Store the valid region
                dominant_emotion = first_face_result['dominant_emotion']
                # We can still extract confidence if needed for display, but it won't affect engagement mapping
                all_emotion_scores = first_face_result['emotion']
                confidence_score = all_emotion_scores.get(dominant_emotion, 0) / 100.0 # Use .get for safety
            else:
                # If box size is unreasonable (too large), treat as no valid detection
                is_valid_detection = False
                # dominant_emotion remains None

    except Exception as e:
        print(f"Error during DeepFace analysis: {e}")
        dominant_emotion = "Error" # Signal an error occurred
        analysis_error = True
        is_valid_detection = False # Ensure no drawing happens

    # --- Determine Engagement State using the new function ---
    # Pass the detected emotion (or None, or "Error") to the mapping function
    current_engagement_state, reason = map_emotion_to_engagement(dominant_emotion)

    # --- Update Not Engaged Timer ---
    # Add the duration of this frame to the total if state is 'Not Engaged'
    if current_engagement_state == 'Not Engaged':
        total_not_engaged_time += delta_time

    # --- Visualization ---

    # Display FPS and Detector info
    cv2.putText(frame, f"FPS: {fps:.1f}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Detector: {detector_backend}", (frame.shape[1]-250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Draw face rectangle and emotion text ONLY if a VALID face was detected and analyzed successfully
    if is_valid_detection and face_region and not analysis_error:
        x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green box for detected face
        # Display emotion and confidence (optional)
        emotion_text = f"{dominant_emotion}: {confidence_score*100:.1f}%"
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display Engagement State
    engagement_text = f"Engagement: {current_engagement_state} {reason}"
    # Color based on engagement state (Green for Engaged, Red for Not Engaged)
    color = (0, 255, 0) if current_engagement_state == "Engaged" else (0, 0, 255)
    cv2.putText(frame, engagement_text, (5, frame.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA) # Moved up slightly

    # Display Total Not Engaged Time
    not_engaged_time_text = f"Total Not Engaged: {total_not_engaged_time:.1f}s"
    cv2.putText(frame, not_engaged_time_text, (5, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA) # Display at bottom

    # Show the final frame
    cv2.imshow('DeepFace Simple Engagement Reader', frame)

    # Update previous frame time *after* all calculations for this frame are done
    prev_frame_time = new_frame_time

    # --- Quit Condition ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Exiting...")
cap.release()
cv2.destroyAllWindows()
print("Resources released.")
print(f"Total time spent in 'Not Engaged' state: {total_not_engaged_time:.2f} seconds")