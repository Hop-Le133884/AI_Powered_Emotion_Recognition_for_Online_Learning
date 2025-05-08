import cv2
import torch
from ultralytics import YOLO

# Configuration
YOLO_MODEL = '../model_yolo/yolov8n.pt'          # YOLOv8 Nano model (COCO dataset)
CONFIDENCE_THRESHOLD = 0.5         # Minimum detection confidence
PROCESS_EVERY_N_FRAMES = 3         # Process every 3rd frame for performance

# Initialize YOLO
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO(YOLO_MODEL).to(device)

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Real-time Person Detection Running. Press 'q' to quit.")

frame_count = 0
last_results = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        continue

    frame_display = frame.copy()
    current_results = []

    # Process every Nth frame
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # YOLO Person Detection
        results = yolo_model.predict(
            frame,
            classes=[0],  # Person class
            conf=CONFIDENCE_THRESHOLD,
            device=device,
            verbose=False
        )

        for box in results[0].boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box
            current_results.append({
                'body_box': (x1, y1, x2, y2)
            })

        last_results = current_results

    # Draw results
    idx_person = 0
    for res in last_results:
        # Draw body box (blue)
        cv2.rectangle(frame_display, 
                      (res['body_box'][0], res['body_box'][1]),
                      (res['body_box'][2], res['body_box'][3]), 
                      (255, 0, 0), 2)
        idx_person += 1
        # Label
        label = f"Person {idx_person}"
        cv2.putText(frame_display, label, 
                    (res['body_box'][0], max(30, res['body_box'][1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


    # Display
    cv2.imshow('Person Detection', frame_display)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()