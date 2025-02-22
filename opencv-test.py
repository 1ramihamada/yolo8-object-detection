from ultralytics import YOLO
import cv2

# Load YOLO model on GPU
model = YOLO("yolov8n.pt").to("cuda")  

# Open the camera at 640x480 resolution for better speed
cap = cv2.VideoCapture(4)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for faster inference
    frame_resized = cv2.resize(frame, (320, 240))  

    # Run YOLO detection with optimized parameters
    results = model(frame_resized, imgsz=320, conf=0.3)

    # Scale factor to map back to original frame size
    scale_x = frame.shape[1] / frame_resized.shape[1]
    scale_y = frame.shape[0] / frame_resized.shape[0]

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            confidence = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID

            # Scale back the bounding box to original frame size
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

            # Draw bounding box
            label = f"{model.names[cls]} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with detections
    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
