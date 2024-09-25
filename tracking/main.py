import os
import time

import cv2
import torch
import yaml
from ultralytics import YOLO
import random

from misc.utils import compute_iou

# Parameters from config.yml file
with open('tracking/config.yml', 'r') as f:
    config = yaml.load(f, yaml.FullLoader)['yolo']

# Device Configuration
device = config['main']['device']
if (device == 'cuda' and not torch.cuda.is_available()): device = 'cpu'

# Visualization Parameters
DISP_INFOS = config['main']['disp_infos']
SAVE_VIDEO = config['main']['save_video']
SAVE_BOUNDING_BOXES = config['main']['save_bounding_boxes']
input_path = config['main']['video_path']
output_path = config['main']['output_path']
output_name = config['main']['output_name']

if(not os.path.exists(output_path)):
    os.makedirs(output_path)
    
# YOLO Parameters
yolo_name = config['model']['yolo_model_name']
tracked_classes = config['model']['tracked_class']
conf_thresh = config['model']['confidence_threshold'] # Set the confidence threshold
model = YOLO(yolo_name + '.pt') # Load the YOLO model
full_dict = {} # Initialize the dictionary to store the results

# Set seed to model
seed = config['main']['seed']
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load the video file
cap = cv2.VideoCapture(R"{}".format(input_path))
video_FPS = cap.get(cv2.CAP_PROP_FPS) # Get the frame rate (FPS) of the video

if(SAVE_VIDEO):
    # Get the width and height of the video frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_path, output_name), fourcc, 30.0, (frame_width, frame_height))

# Initialize dictionary to hold the previous frame's bounding boxes, classes, confidences, and cropped images
full_dict = {}

# Initialize frame counter
frame_number = 0

# Run the tracking loop
while cap.isOpened():

    # Read the image frame from data source
    success, img = cap.read()
    if not success: break

    # Increment frame counter
    frame_number += 1

    # Calculate the timestamp for the current frame
    timestamp = frame_number / video_FPS

    # YOLO Inference
    start_time_yolo = time.perf_counter()

    # Run YOLO tracking on the frame
    results = model.track(img, classes=list(tracked_classes.keys()), conf=conf_thresh, persist=True)
    
    end_time_yolo = time.perf_counter()

    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    if(results[0].boxes.id is not None):
        ids = results[0].boxes.id.tolist()
    else:
        ids = [0] * len(classes)
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    # Iterate through the results
    current_frame_data = []

    for box, cls, id, conf in zip(boxes, classes, ids, confidences):
            x1, y1, x2, y2 = map(int, box)  # Convert to integers
            confidence = conf
            detected_class = int(cls)
            name = names[int(cls)]
            id = int(id)
            crop = img[y1:y2, x1:x2]  # Crop the bounding box from the image

            # Create an empty list if it doesn't exist
            if (id not in full_dict):
                full_dict[id] = []

            # We need a way to check the current object coordinates against all the previous objects
            # For doing this, we need to filter the dictionary with bounding boxes taken with a timestamp lower than the current one
            # with a certain threshold (e.g. I want to take into account only the objects detected in the last 10 seconds)
            # This way we can compare the current object with the previous ones and check if they are the same object
            # If they are the same object (also thanks to the IoU), we can update the row by moving it to the appropiate ID key

            # Check first if full_dict has at least a key and a value
            keys = list(full_dict.keys())
            if(len(keys) > 0) and (len(full_dict[keys[0]]) > 0):      
                filtered_dict = {k: v for k, v in full_dict.items() if timestamp - v['timestamp'] < 10}
            else:
                filtered_dict = {}

            # Initialize the flag to check if the object has been found in the previous frame
            found = False

            # Iterate through the filtered dictionary
            for key, value in filtered_dict.items():
                # Compute the IoU between the current object and the previous one
                iou = compute_iou(box, value['bounding_box'])

                # If the IoU is greater than 0.85, we can assume that the current object is the same as the previous one
                if iou > 0.85:
                    # Update the ID of the current object
                    id = key
                    # Update the flag
                    found = True
                    # Break the loop
                    break

            full_dict[id].append({
                'class': name,
                'confidence': confidence,
                'bounding_box': box,
                'cropped_image': crop,
                'timestamp': timestamp
            })

            # Visualize the bounding box on the image
            if SAVE_BOUNDING_BOXES:
                filename = os.path.join(output_path, f"{name}_ID-{id}_{confidence:.2f}.jpg")
                cv2.imwrite(filename, crop)

    # Annotate and show the image
    annotated_frame = results[0].plot(font_size=15, line_width=3)

    # Display the FPS and number of detected objects
    if DISP_INFOS:
        total_time = end_time_yolo - start_time_yolo
        fps = 1 / total_time
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(annotated_frame, f'DETECTED OBJECTS: {len(results[0].boxes.cls)}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(annotated_frame, f'TRACKED CLASS: {list(tracked_classes.values())}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(annotated_frame, f'FRAME NUMBER: {frame_number}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Save the annotated frame to the output video
    if SAVE_VIDEO:
        out.write(annotated_frame)

    # Display the annotated frame
    cv2.imshow("YOLO Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break