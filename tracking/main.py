import cv2
import time
import torch
import yaml

from detector import YOLOv5
from tracker import DeepSortTracker

# Parameters from config.yml file
with open('tracking/config.yml', 'r') as f:
    config = yaml.load(f, yaml.FullLoader)['yolov5_deepsort']

# Visualization Parameters
DISP_INFOS = config['main']['disp_infos']

# CUDA for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the YOLOv5 object detector and DeepSORT deepsort_tracker
yolo_detector = YOLOv5(config, device=device)
deepsort = DeepSortTracker(config)

# Load the video file
cap = cv2.VideoCapture(R"{}".format(config['main']['video_path']))

track_history = {}    # Define a empty dictionary to store the previous center locations for each track ID

while cap.isOpened():

    success, img = cap.read() # Read the image frame from data source 
 
    start_time = time.perf_counter() # Start Timer - needed to calculate FPS
    
    # Object Detection
    results = yolo_detector.run(img)  # Run the YOLOv5 object detector 
    detections, num_objects = yolo_detector.extract_detections(results, img, height=img.shape[0], width=img.shape[1]) # Plot the bounding boxes and extract detections (needed for DeepSORT) and number of relavent objects detected

    # Object Tracking
    tracks_current = deepsort.object_tracker.update_tracks(detections, frame=img)
    deepsort.display_track(track_history, tracks_current, img)
    
    # FPS Calculation
    end_time = time.perf_counter()
    total_time = end_time - start_time
    fps = 1 / total_time

    # Descriptions on the output visualization
    if DISP_INFOS:
        cv2.putText(img, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img, f'DETECTED OBJECTS: {num_objects}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img, f'TRACKED CLASS: {yolo_detector.tracked_class}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()