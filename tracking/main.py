import cv2
import time
import yaml
import os

from ultralytics import YOLO

# Parameters from config.yml file
with open('tracking/config.yml', 'r') as f:
    config = yaml.load(f, yaml.FullLoader)['yolov8']

# Visualization Parameters
DISP_INFOS = config['main']['disp_infos']
SAVE_VIDEO = config['main']['save_video']
input_path = config['main']['video_path']
output_path = config['main']['output_path']
output_name = config['main']['output_name']

if(not os.path.exists(output_path)):
    os.makedirs(output_path)
    
# YOLOv8 Parameters
yolo_name = config['model']['yolo_model_name']
tracked_classes = config['model']['tracked_class']
conf_thresh = config['model']['confidence_threshold'] # Set the confidence threshold
model = YOLO(yolo_name + '.pt') # Load the YOLOv8 model

# Load the video file
cap = cv2.VideoCapture(R"{}".format(input_path))

if(SAVE_VIDEO):
    # Get the width and height of the video frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Define the codec and create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_path, output_name), fourcc, 30.0, (frame_width, frame_height))

# Run the tracking loop
while cap.isOpened():

    # Read the image frame from data source
    success, img = cap.read()
    if not success: break

    start_time = time.perf_counter() # Start Timer - needed to calculate FPS

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(img, classes=list(tracked_classes.keys()), conf=conf_thresh, persist=True)
    
    # for r in results:
    #     print(r.boxes.id)  # print tracking IDs

    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    # FPS Calculation
    end_time = time.perf_counter()
    total_time = end_time - start_time
    fps = 1 / total_time

    # Descriptions on the output visualization
    if DISP_INFOS:
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(annotated_frame, f'DETECTED OBJECTS: {len(results[0].boxes.cls)}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(annotated_frame, f'TRACKED CLASS: {list(tracked_classes.values())}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Write the frame with bounding boxes to the video file if SAVE_VIDEO is True
    if SAVE_VIDEO: out.write(annotated_frame)
    
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27: break

# Release and destroy all windows before termination
cap.release()
if SAVE_VIDEO: out.release()
cv2.destroyAllWindows()