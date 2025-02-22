# Real-Time Object Detection with YOLOv8

## Overview
This Python script utilizes the YOLOv8 object detection model to perform real-time object detection from a camera feed. It loads a YOLO model onto a GPU, processes the video feed for object detection, and displays bounding boxes with labels and confidence scores on detected objects.

## Features
- Uses **YOLOv8** for object detection.
- Runs on **GPU** for faster inference.
- Processes frames at **320x240 resolution** for speed optimization.
- Displays real-time **bounding boxes** with class labels and confidence scores.
- Allows **quitting** the program by pressing 'q'.

## Requirements
- Python 3.x
- OpenCV (`pip install opencv-python`)
- Ultralytics YOLO (`pip install ultralytics`)
- CUDA-compatible GPU for acceleration (optional but recommended)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/yolo-camera-detection.git
   cd yolo-camera-detection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Ensure that your camera is properly connected and detected by OpenCV.

## Usage
Run the script to start real-time object detection:
```sh
python yolo_camera_detection.py
```

## How It Works
1. **Load YOLO Model**
   ```python
   model = YOLO("yolov8n.pt").to("cuda")
   ```
   The model is loaded onto the GPU for faster inference.

2. **Open Camera Stream**
   ```python
   cap = cv2.VideoCapture(4)
   cap.set(3, 640)  # Width
   cap.set(4, 480)  # Height
   ```
   The camera stream is opened at 640x480 resolution.

3. **Process Each Frame**
   - Frames are resized to 320x240 for faster processing.
   - YOLO model runs inference on each frame.
   - Bounding boxes are scaled back to the original frame size.
   - Detected objects are labeled and displayed.

4. **User Interaction**
   - The detection window displays objects in real-time.
   - Press **'q'** to exit the application.

## Example Output
```
Object Detected: Person 0.85
Object Detected: Chair 0.78
```

## Error Handling
- Handles **camera connection failures**.
- Ensures **graceful exit** on user interruption.
- Properly releases resources before closing.

## License
This project is licensed under the MIT License.

