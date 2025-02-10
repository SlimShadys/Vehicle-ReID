import gc
import os
import pickle
import random
import sys
from collections import defaultdict

import cv2
import imageio as imageio
import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Add the parent directory of 'tracking' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import _C as cfg_file
from misc.printer import Logger
from reid.models.color_model import SVM, EfficientNet
from pipeline import load_models_and_transformations

# Function to draw bounding boxes and tracks
def draw_box_and_tracks(frame, boxes, ids, colors, track_history):
    TINT_COLOR = (0, 0, 0)  # Black
    TRANSPARENCY = .50  # Degree of transparency, 0-100%
    OPACITY = int(255 * TRANSPARENCY)

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    overlay = Image.new('RGBA', img_pil.size, TINT_COLOR+(0,))
    draw_overlay = ImageDraw.Draw(overlay, "RGBA")  # Create a context for drawing things on it.

    for box, track_id in zip(boxes, ids):

        if track_id == -1: continue # Skip the object if the tracker does not detect it

        # Convert TLWH to XYXY format
        x0, y0, w, h = box
        x1 = x0 + w
        y1 = y0 + h

        # Draw tracking lines (track history)
        track = track_history.get(track_id, [])

        # Vehicle Color Prediction
        # Out of all the colors present in the track history, get the most frequent color
        id_color = colors[track_id]
        car_color = track[-1][1] if len(track) > 0 else "Unknown"

        # Draw bounding box
        draw.rectangle([x0, y0, x1, y1], outline=id_color, width=3)
        
        text = f"ID: {track_id},\nCOLOR: {car_color}"
        textcoords = draw_overlay.multiline_textbbox((x0, y1), text, font=font)

        if textcoords[3] >= img_pil.size[1] - 400:
            y1 = y0 - (textcoords[3] - textcoords[1])

        # draw rectangle in the background
        coords = draw_overlay.multiline_textbbox((x0, y1), text, font=font)

        # add some padding
        textcoords = (coords[0] - 2, coords[1] - 2, coords[2] + 2, coords[3] + 2)
        draw_overlay.rectangle(textcoords, fill=id_color + (OPACITY,))

        # draw the text finally
        draw_overlay.multiline_text((x0, y1), text, (0, 0, 0), font=font)

        if len(track) > 1:  # Ensure there are at least 2 points in the track history to draw a line
            # Convert track history to a list of tuples
            points = [(int(p[0]), int(p[1])) for p, _ in track]
            
            # Draw the polyline directly using PIL's line() method
            draw.line(points, fill=id_color, width=4)

    # Combine the overlay and original image
    img_pil = Image.alpha_composite(img_pil.convert("RGBA"), overlay)

    # Convert the final PIL image back to a NumPy array for OpenCV compatibility
    return np.array(img_pil)

# Free resources
gc.collect()
torch.cuda.empty_cache()

# Number of classes in each dataset (Only for ID classification tasks, hence only training set is considered)
num_classes = {
    'ai_city': 440,
    'ai_city_mix': 1802,
    'ai_city_sim': 1362,
    'vehicle_id': 13164,
    'veri_776': 576,
    'veri_wild': 30671,
    'vric': 2811,
    'vru': 7086,
}

# Various configs
misc_cfg = cfg_file.MISC
reid_cfg = cfg_file.REID
tracking_cfg = cfg_file.TRACKING
tracker_cfg = cfg_file.TRACKER
yolo_cfg = cfg_file.YOLO
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
test_name = misc_cfg.TEST_NAME
# ============================================================ #

# Load the models and transformations
_, model, color_model, transforms = load_models_and_transformations(device, cfg_file)

# ========================== LOAD YOLO MODEL ========================== #
# Define a font for drawing text
font = ImageFont.load_default(size=12)

pickle_global_path = cfg_file.MTMC.PICKLE_GLOBAL_PATH

# Filter pickle files by only the ones starting with "multitracklet"
pickle_files = [os.path.join(pickle_global_path, test_name, f) for f in os.listdir(os.path.join(pickle_global_path, test_name)) if os.path.isfile(os.path.join(pickle_global_path, test_name, f)) and f.startswith("multitracklet")]

all_frames = {}

df = pd.DataFrame(columns=['ID', 'frame_number', 'bounding_box', 'features', 'camera_id'])

# get total number of samples in the pickle files
total_n_samples = 0
for pickle_file in pickle_files:
    with open(pickle_file, "rb") as f:
        res = pickle.load(f)
        for r in res:
            total_n_samples += len(r['frame_numbers'])
        
with tqdm(total=total_n_samples, desc="Processing pickle files", unit="Vehicle") as pbar:        
    for i, pickle_file in enumerate(pickle_files): # PICKLE SINGLE FILES
        with open(pickle_file, "rb") as f:
            res = pickle.load(f)
            
            # We must have the same schema as get_all_frames() from the Database
            for r in res:           
                for frame_num, bbox, feature in zip(r['frame_numbers'], r['bboxes'], r['features']):
                    # df = df.append({'ID': vehicle_id, 'frame_number': frame_num, 'bounding_box': bbox, 'features': feature, 'camera_id': i}, ignore_index=True)
                    df = pd.concat([pd.DataFrame([[r['id'], frame_num, bbox, feature, i]], columns=df.columns), df], ignore_index=True)
                    pbar.update(1)
                
# We must create a set of IDs contained in the all_frames dictionary
all_ids = sorted(df['ID'].unique())
import random
random_colors = [(int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)) for _ in range(len(all_ids))]

# assign a color to each ID in the Dataframe
df['color'] = df['ID'].apply(lambda x: random_colors[all_ids.index(x)])

# Retrieve the color column from the DataFrame and convert it to a dict with the corresponding ID
colors = dict(zip(df['ID'], df['color']))

# ========================== LOAD COLOR MODEL ========================== #
model_configs = reid_cfg.MODEL
model_name = reid_cfg.MODEL.NAME
pretrained = reid_cfg.MODEL.PRETRAINED
dataset_name = reid_cfg.DATASET.DATASET_NAME
color_model_configs = reid_cfg.COLOR_MODEL
model_val_path = reid_cfg.TEST.MODEL_VAL_PATH

# HARD - CODED!
input_paths = [('./data/AICity22_Track1_MTMC_Tracking/validation/S02/c006/vdo.avi', 0),
               ('./data/AICity22_Track1_MTMC_Tracking/validation/S02/c007/vdo.avi', 1),]

for (video, cam_id) in input_paths:
    # Check if the path exists
    if not os.path.exists(video):
        raise FileNotFoundError(f"The path '{video}' does not exist.")

    if os.path.isfile(video): # We are working with a video
        video_extensions = {".mp4", ".avi", ".mov", ".mkv"}  # Add other common video extensions as needed
        file_extension = os.path.splitext(video)[-1].lower()
        if file_extension not in video_extensions:
            logger.error(f"Invalid video file format. Supported formats are {video_extensions}")
            raise Exception()
        else:
            logger.info(f"Input path is a video file. Loading video from {video}")
    else:
        logger.error(f"Invalid input path. Please provide a valid path to a video file or a directory containing frames.")
        raise Exception()

    # Get the video props using OpenCV
    cap = cv2.VideoCapture(R"{}".format(video))
    if not cap.isOpened():
        logger.info(f"Error opening video file at {video}")
        raise Exception()
    else:
        logger.info(f"Successfully opened video file at {video}")

    total_n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get the total number of frames in the video
    FPS = cap.get(cv2.CAP_PROP_FPS)                   # Get the frame rate (FPS) of the video
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # Get the width of the frame
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the height of the frame

    # Load up the video using imageio
    reader = iio.imopen(video, io_mode='r', plugin='pyav')
    codec = reader.metadata()['codec']
    reader = reader.iter()
        
    if SAVE_VIDEO:
        # Check if output path exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        writer = imageio.get_writer(os.path.join(output_path, video.split('/')[-1] + f'_cam{cam_id}' + f'_tracked.{extension}'),
                                    format='FFMPEG', mode='I',
                                    fps=FPS, codec=codec, macro_block_size=8)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Run the tracking loop
    with tqdm(total=total_n_frames, desc="Processing Video", unit="frame") as pbar:
        # Initialize frame number and start the loop based on input type
        frame_number = 0

        # Run the tracking loop
        while frame_number < total_n_frames:

            # Load the next frame from the video
            try:
                orig_frame = next(reader)
            except StopIteration:
                break  # End of video stream
            
            frame_number += 1 # Increment frame counter

            if frame_number == 200: # For testing purposes
                break

            # Calculate the timestamp for the current frame
            timestamp = frame_number / FPS

            # Extract bounding boxes and corresponding IDs from the DataFrame for this frame_number and camera_id
            bboxes = df[(df['frame_number'] == frame_number) & (df['camera_id'] == cam_id)]['bounding_box'].values
            ids = df[(df['frame_number'] == frame_number) & (df['camera_id'] == cam_id)]['ID'].values
            
            # Correctly update the track history with center coordinates
            for box, track_id in zip(bboxes, ids):
                if track_id == -1:
                    continue # Skip the object if the tracker does not detect it
                else:
                    track = track_history[track_id]
                    
                    # Convert TLWH to XYXY format
                    x0, y0, w, h = map(int, box)
                    x1 = x0 + w
                    y1 = y0 + h
                                       
                    # Append the center of the bounding box to the track history
                    center_x = x0 + ((x1 - x0) / 2)
                    center_y = y0 + ((y1 - y0) / 2)

                    x = Image.fromarray(orig_frame[y0:y1, x0:x1])                                   # Convert to PIL Image
                    #x.save(f"./tracking/{frame_number}_{track_id}.jpg")
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
                    else:
                        color_prediction = "Unknown"

                    # Append the center of the bounding box and the color prediction to the track history
                    track.append(((center_x, center_y), color_prediction))
                    
                    # Keep only the last 20 points
                    if len(track) > 20:
                        track.pop(0)

            # Draw boxes and tracklets
            annotated_frame = draw_box_and_tracks(orig_frame, bboxes, ids, colors, track_history)

            # Display the FPS and number of detected objects
            if DISP_INFOS:
                
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
                    f'DETECTED OBJECTS: {len(bboxes)}',
                    f'FRAME NUMBER: {frame_number}',
                    f'TIMESTAMP: {timestamp:.2f} seconds'
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
            #window_name = f"MTMC Test Tracking"
            #cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            # cv2.moveWindow(window_name, 900, 900)
            #cv2.imshow(window_name, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

            # Break the loop if 'q' is pressed
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break

            pbar.update(1)
        
    # Release the video reader and writer
    if SAVE_VIDEO: writer.close()
        
    cv2.destroyAllWindows()
