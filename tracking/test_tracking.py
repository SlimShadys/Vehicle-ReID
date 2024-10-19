import gc
import os
import random
import sys
import time
from collections import defaultdict

import cv2
import imageio as imageio
import imageio.v3 as iio
import numpy as np
import torch
from matplotlib import pyplot as plt
from model import load_yolo
from PIL import Image, ImageDraw, ImageFont

# Add the parent directory of 'tracking' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import _C as cfg_file
from misc.printer import Logger
from misc.utils import create_mask
from reid.datasets.transforms import Transformations
from reid.model import ModelBuilder
from reid.models.color_model import EfficientNet, SVM

# Function to draw bounding boxes and tracks
def draw_box_and_tracks(frame, boxes, ids, classes, confidences, track_history):
    TINT_COLOR = (0, 0, 0)  # Black
    TRANSPARENCY = .50  # Degree of transparency, 0-100%
    OPACITY = int(255 * TRANSPARENCY)

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    overlay = Image.new('RGBA', img_pil.size, TINT_COLOR+(0,))
    draw_overlay = ImageDraw.Draw(overlay, "RGBA")  # Create a context for drawing things on it.

    for box, track_id, cls, conf in zip(boxes, ids, classes, confidences):

        x0, y0, x1, y1 = box # Format: XYXY

        color = tuple(int(c * 255) for c in colors[track_id % len(colors)])

        # Draw tracking lines (track history)
        track = track_history.get(track_id, [])

        # Vehicle Color Prediction
        # Out of all the colors present in the track history, get the most frequent color
        car_color = track[-1][1] if len(track) > 0 else "Unknown"

        # Draw bounding box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        text = f"ID: {track_id}, {str(names[int(cls)]).upper()}: {conf:.2f}\nCOLOR: {car_color}"

        textcoords = draw_overlay.multiline_textbbox((x0, y1), text, font=font)

        if textcoords[3] >= img_pil.size[1] - 400:
            y1 = y0 - (textcoords[3] - textcoords[1])

        # draw rectangle in the background
        coords = draw_overlay.multiline_textbbox((x0, y1), text, font=font)

        # add some padding
        textcoords = (coords[0] - 2, coords[1] - 2, coords[2] + 2, coords[3] + 2)
        draw_overlay.rectangle(textcoords, fill=color + (OPACITY,))

        # draw the text finally
        draw_overlay.multiline_text((x0, y1), text, (0, 0, 0), font=font)

        if len(track) > 1:  # Ensure there are at least 2 points in the track history to draw a line
            # Convert track history to a list of tuples
            points = [(int(p[0]), int(p[1])) for p, _ in track]
            
            # Draw the polyline directly using PIL's line() method
            draw.line(points, fill=color, width=4)

    # Combine the overlay and original image
    img_pil = Image.alpha_composite(img_pil.convert("RGBA"), overlay)

    # Convert the final PIL image back to a NumPy array for OpenCV compatibility
    return np.array(img_pil)

# Free resources
gc.collect()
torch.cuda.empty_cache()

# Number of classes in each dataset (Only for ID classification tasks, hence only training set is considered)
num_classes = {
    'veri_776': 576,
    'veri_wild': 30671,
    'vehicle_id': 13164,
    'vru': 7086,
}

# Various configs
misc_cfg = cfg_file.MISC
reid_cfg = cfg_file.REID
tracking_cfg = cfg_file.TRACKING
db_cfg = cfg_file.DB

# Device Configuration
device = misc_cfg.DEVICE
if device == 'cuda' and torch.cuda.is_available():
    device = device + ":" + str(misc_cfg.GPU_ID)
else:
    device = 'cpu'

# Set seed
seed = misc_cfg.SEED
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load Logger
logger = Logger()

# ========================== MISC CONFIGS ========================== #
SAVE_VIDEO = tracking_cfg.SAVE_VIDEO
DISP_INFOS = tracking_cfg.DISP_INFOS
output_path = tracking_cfg.OUTPUT_PATH
extension = tracking_cfg.VIDEO_PATH.split('.')[-1]
use_roi_mask = tracking_cfg.USE_ROI_MASK
roi_mask_path = tracking_cfg.ROI_MASK_PATH

# ========================== LOAD YOLO MODEL ========================== #
yolo_model = load_yolo(tracking_cfg)
input_path = tracking_cfg.VIDEO_PATH
tracked_classes = tracking_cfg.MODEL.TRACKED_CLASS
conf_thresh = tracking_cfg.MODEL.CONFIDENCE_THRESHOLD
yolo_img_size = tracking_cfg.MODEL.YOLO_IMAGE_SIZE
yolo_iou_threshold = tracking_cfg.MODEL.YOLO_IOU_THRESHOLD
use_agnostic_nms = tracking_cfg.MODEL.USE_AGNOSTIC_NMS
tracker_yaml_file = tracking_cfg.MODEL.YOLO_TRACKER

# ========================== LOAD COLOR MODEL ========================== #
model_configs = reid_cfg.MODEL
model_name = reid_cfg.MODEL.NAME
pretrained = reid_cfg.MODEL.PRETRAINED
dataset_name = reid_cfg.DATASET.DATASET_NAME
color_model_configs = reid_cfg.COLOR_MODEL
model_val_path = reid_cfg.TEST.MODEL_VAL_PATH

# If the color model is SVM, we need to Re-ID features from the ReID model first
if color_model_configs.NAME == 'svm':
    logger.reid(f"Building {reid_cfg.MODEL.NAME} model...")
    model_builder = ModelBuilder(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes[dataset_name],
        model_configs=model_configs,
        device=device
    )

    # Get the model and move it to the device
    model = model_builder.move_to(device)

    # Load parameters from a .pth file
    if (model_val_path != ""):
        logger.reid("Loading model parameters from file...")
        if ('ibn' in model_name):
            model.load_param(model_val_path)
        else:
            checkpoint = torch.load(model_val_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
        logger.reid(f"Successfully loaded model parameters from file: {model_val_path}")

    model.eval()
    print(model)

# Build the color model
logger.color(f"Loading color model from {color_model_configs.NAME}...")
model_builder = ModelBuilder(
    model_name=color_model_configs.NAME,
    pretrained=pretrained,
    num_classes=num_classes[dataset_name],
    model_configs=color_model_configs,
    device=device
)
color_model = model_builder.move_to(device)
logger.color(f"Successfully loaded color model from {color_model_configs.NAME}!")

# Transformations needed for the ReID model
transforms = Transformations(dataset=None, configs=reid_cfg.AUGMENTATION)
# ====================================================================== #

# Load the video file and get its metadata
reader = iio.imopen(input_path, 'r', plugin='pyav')
codec = reader.metadata()['codec']  # Get the Codec used in the video
fps = reader.metadata()['fps'] # Get the FPS of the video
height, width, channels = iio.imread(input_path, index=0, plugin="pyav").shape # Get the shape of the video

if SAVE_VIDEO:
    writer = imageio.get_writer(os.path.join(output_path, input_path.split('/')[-1] + '_tracked.' + extension),
                                format='FFMPEG', mode='I',
                                fps=fps, codec=codec, macro_block_size=8)

# Store the track history
track_history = defaultdict(lambda: [])

# Color mapping for different track IDs
colors = plt.get_cmap("tab20").colors

# Define a font for drawing text
font = ImageFont.load_default(size=12)

time_array = []

if (use_roi_mask == True):
    # Load the mask
    if (roi_mask_path != ""):
        mask = cv2.imread(roi_mask_path, 0)  # Grayscale mask
    else:
        # We need to extract a single frame from the video to create the mask
        mask_frame = iio.imread(input_path, index=0, plugin="pyav")
        mask = create_mask(mask_frame, input_path)

    mask = cv2.resize(mask, (width, height))

# Run the tracking loop
for frame_number, orig_frame in enumerate(reader.iter()):

    # Calculate the timestamp for the current frame
    timestamp = frame_number / fps

    # YOLO Inference
    start_time_yolo = time.perf_counter()

    # Apply the mask: Keep only the region inside the mask (ROI)
    if (use_roi_mask == True):
        roi_frame = cv2.bitwise_and(orig_frame, orig_frame, mask=mask)
        frame = roi_frame
        #cv2.imshow("ROI Frame", cv2.cvtColor(roi_frame, cv2.COLOR_RGB2BGR))
    else:
        frame = orig_frame

    # Run YOLO tracking on the frame
    results = yolo_model.track(frame, classes=list([idx for idx, _ in tracked_classes]), conf=conf_thresh,
                                imgsz=yolo_img_size, iou=yolo_iou_threshold,
                                agnostic_nms=use_agnostic_nms, tracker=tracker_yaml_file,
                                device=device, persist=True, verbose=False)
    
    end_time_yolo = time.perf_counter()

    time_array.append(end_time_yolo - start_time_yolo)

    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.cpu().tolist()
    classes = results[0].boxes.cls.cpu().tolist()
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().cpu().tolist()
    else:
        ids = [0] * len(classes)
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    # Correctly update the track history with center coordinates
    for box, track_id in zip(boxes, ids):
        x0, y0, x1, y1 = map(int, box) # Format: XYXY
        track = track_history[track_id]
        # Append the center of the bounding box to the track history
        center_x = x0 + ((x1 - x0) / 2)
        center_y = y0 + ((y1 - y0) / 2)

        x = Image.fromarray(orig_frame[y0:y1, x0:x1])                                   # Convert to PIL Image
        x.save(f"./tracking/{frame_number}_{track_id}.jpg")
        tensor_x = transforms.val_transform(x)                                          # Convert to torch.Tensor
        tensor_x = tensor_x.unsqueeze(0).float().to(device)                             # Convert to Tensor and send to device

        # EfficientNet
        if isinstance(color_model, EfficientNet):
            prediction = color_model.predict(tensor_x)
            color_prediction = [entry['color'] for entry in prediction]
        # ResNet + SVM
        elif isinstance(color_model, SVM):
            svm_embedding = model(tensor_x).detach().cpu().numpy()
            color_prediction = color_model.predict(svm_embedding)

        # Append the center of the bounding box and the color prediction to the track history
        track.append(((center_x, center_y), color_prediction))
        
        # Keep only the last 20 points
        if len(track) > 20:
            track.pop(0)

    # Draw boxes and tracklets
    annotated_frame = draw_box_and_tracks(orig_frame, boxes, ids, classes, confidences, track_history)

    # Display the FPS and number of detected objects
    if DISP_INFOS:
        total_time = end_time_yolo - start_time_yolo
        fps = 1 / total_time
        
        # Define colors and font size for the text
        info_color = (255, 255, 255)  # White text color
        background_color = (0, 0, 0)  # Black background for text box
        background_opacity = 0.6  # Background transparency
        padding = 10  # Padding for text box

        # Define font size for better visibility
        font_size = 0.60
        font_thickness = 1

        # Information to display
        info_text = [
            #f'FPS: {int(fps)}',
            f'DETECTED OBJECTS: {len(results[0].boxes.cls)}',
            #f'TRACKED CLASS: {list(tracked_classes.values())}',
            f'FRAME NUMBER: {frame_number}'
        ]

        # Starting y-position for the info display
        y_start = 40

        for idx, text in enumerate(info_text):
            # Calculate the width and height of the text box
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)

            # Create a filled rectangle as the background of the text
            cv2.rectangle(
                annotated_frame, 
                (20, y_start - text_height - padding), 
                (20 + text_width + padding * 2, y_start + padding), 
                (background_color[0], background_color[1], background_color[2], int(background_opacity * 255)),
                -1
            )

            # Draw the text with the font on top of the rectangle
            cv2.putText(
                annotated_frame, text, 
                (20 + padding, y_start), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_size, 
                info_color, 
                font_thickness
            )

            # Move to the next line
            y_start += text_height + 2 * padding

    if SAVE_VIDEO:
        writer.append_data(annotated_frame)

    # Display the annotated frame
    cv2.imshow(f"{tracking_cfg.MODEL.YOLO_MODEL_NAME} Tracking", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
# Release the video reader and writer
reader.close()

if SAVE_VIDEO: writer.close()
    
cv2.destroyAllWindows()

# Print the mean time taken
print(f"Mean time taken for YOLO Inference: {np.mean(time_array):.4f} seconds")
