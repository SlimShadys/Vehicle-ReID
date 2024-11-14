# In the pipeline file, we might need to:
# - Load the YOLO Engine for Object Detection and Tracking tasks                -> Done
#   -> Load the YOLO model and weights                                          -> Done
# - Load the ReID Engine for Reidentification tasks                             -> Done
#   -> Load the ReID model and weights                                          -> Done
# - Load the Spectrico Car Recognition Engine for Car Color Recognition tasks   -> Done
#   -> Load the Spectrico Car Recognition model and weights                     -> Done
# - Load the video file from which we want to extract the frames                -> Done
#   -> Load the video file                                                      -> Done
# - Filter out unwanted frames based on the color of the car                    -> Done
#   -> Run the color model on the frames                                        -> Done
#   -> Filter out the frames based on the color                                 -> Done
#   -> Filter out the stationary vehicles                                       -> Done
#   -> Filter out incomplete Bounding Boxes                                     -> Done
# - Run the ReID model on the remaining frames                                  -> Done
# - Insert the data into the MongoDB database                                   -> Done
#   -> Insert the vehicles into the database                                    -> Done
#   -> Insert the trajectories into the database                                -> Done
#   -> Insert the bounding boxes into the database                              -> Done
#   -> Insert the cameras into the database                                     -> Done
# - Save the bounding boxes of the vehicles                                     -> Done
# - Print a summary of the filtering                                            -> Done
# - Release the video capture and destroy all OpenCV windows                    -> Done
# - Print a message to indicate the end of the pipeline                         -> Done
# - Update trajectories, bounding boxes, and vehicles in the database           -> 90% (final testing needed)
# - Create Graphs for the trajectories                                          -> 90%
# - Create Target Search Engine                                                 -> 0%
# - Create a GUI for the pipeline                                               -> 0%

import argparse
from datetime import datetime
import gc
import os
import re
import shutil

import bson
import cv2
import imageio.v3 as iio
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
import yaml

from config import _C as cfg_file
from db.database import Database
from graph.model import GraphEngine
from misc.printer import Logger
from misc.utils import (compare_trajectories, compress_img,
                        cosine_similarity_np, create_mask, decompress_img,
                        euclidean_dist, extract_numeric_id, load_middle_frame,
                        remove_below_threshold, sample_trajectory_frames,
                        set_seed)
from reid.datasets.transforms import Transformations
from reid.model import ModelBuilder
from tracking.filter import filter_frames
from tracking.model import load_yolo
from tracking.unification import get_max_similarity_per_vehicle
from misc.timer import PerformanceTimer

# Various configs
misc_cfg = cfg_file.MISC
reid_cfg = cfg_file.REID
tracking_cfg = cfg_file.TRACKING
db_cfg = cfg_file.DB
num_classes = {
    'veri_776': 576,
    'veri_wild': 30671,
    'vehicle_id': 13164,
    'vru': 7086,
}

# SIMILARITY_THRESHOLD
similarity_threshold = 0.60 # 0.69
removing_threshold = 0.60
comparing_threshold = 0.60

def load_models_and_transformations(device, cfg):
    """Load YOLO, ReID model, color model, and transformations."""
    # Load YOLO model
    yolo_model = load_yolo(cfg.TRACKING)

    # Load ReID Model
    dataset_name = reid_cfg.DATASET.DATASET_NAME
    model_builder = ModelBuilder(
        model_name=cfg.REID.MODEL.NAME,
        pretrained=cfg.REID.MODEL.PRETRAINED,
        num_classes=num_classes[dataset_name],
        model_configs=cfg.REID.MODEL,
        device=device
    )
    model = model_builder.move_to(device)
    model.eval()

    # Load Color Model
    car_classifier = ModelBuilder(
        model_name=cfg.REID.COLOR_MODEL.NAME,
        pretrained=cfg.REID.MODEL.PRETRAINED,
        num_classes=num_classes[dataset_name],
        model_configs=cfg.REID.COLOR_MODEL,
        device=device
    ).move_to(device)
    
    # Load Transformations
    transforms = Transformations(dataset=None, configs=cfg.REID.AUGMENTATION)
    return yolo_model, model, car_classifier, transforms

def load_config_and_device(config_path=None):
    """Load config, initialize device and set seed."""
    if config_path:
        cfg_file.merge_from_file(config_path)
    device = cfg_file.MISC.DEVICE
    if device == 'cuda' and torch.cuda.is_available():
        device = f"{device}:{cfg_file.MISC.GPU_ID}"
    else:
        device = 'cpu'
    set_seed(cfg_file.MISC.SEED)
    return cfg_file, device

def setup_database():
    """Initialize the database connection."""
    logger = Logger()
    db = Database(db_configs=cfg_file.DB, logger=logger)
    if cfg_file.DB.CLEAN_DB:
        logger.debug("Cleaning the database...")
        db.clean()
    return db

def run_mtsc(yolo_model, model, car_classifier, transforms, device, camera_data):
    """Run the pipeline for a single camera, used in both MTSC and MTMC."""
    # Load Logger
    logger = Logger()

    # Load PerformanceTimer
    perf_timer = PerformanceTimer()

    # DB connection
    USE_DB = db_cfg.USE_DB
    reid_DB = Database(db_configs=db_cfg, logger=logger) if USE_DB else None

    # Get the similarity method to use
    similarity_algorithm    = reid_cfg.TEST.SIMILARITY_ALGORITHM    # 'cosine' / 'euclidean'
    similarity_method       = reid_cfg.TEST.SIMILARITY_METHOD       # 'individual' / 'mean'

    # If GT is available, we must run MOTA metrics
    cfg_file.TRACKING.GT = camera_data.get('gt', None)
    if cfg_file.TRACKING.GT:
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = f'predictions_camera-{camera_data["_id"]}_{date}.txt'
        res = {
            "frame": [],
            "track_id": [],
            "bbox_topleft_x": [],
            "bbox_topleft_y": [],
            "bbox_width": [],
            "bbox_height": [],
        }

    if misc_cfg.RUN_PIPELINE:
        # Configure camera-specific parameters
        cfg_file.TRACKING.VIDEO_PATH = camera_data['path']
        cfg_file.TRACKING.ROI_MASK_PATH = camera_data.get('roi', None)

        # ========================== YOLO ========================== #
        input_path          = tracking_cfg.VIDEO_PATH
        tracked_classes     = tracking_cfg.MODEL.TRACKED_CLASS
        conf_thresh         = tracking_cfg.MODEL.CONFIDENCE_THRESHOLD
        yolo_img_size       = tracking_cfg.MODEL.YOLO_IMAGE_SIZE
        yolo_iou_threshold  = tracking_cfg.MODEL.YOLO_IOU_THRESHOLD
        use_agnostic_nms    = tracking_cfg.MODEL.USE_AGNOSTIC_NMS
        tracker_yaml_file   = tracking_cfg.MODEL.YOLO_TRACKER
        use_roi_mask        = tracking_cfg.USE_ROI_MASK
        roi_mask_path       = tracking_cfg.ROI_MASK_PATH
        input_path          = cfg_file.TRACKING.VIDEO_PATH
        tracked_classes     = cfg_file.TRACKING.MODEL.TRACKED_CLASS
        
        # Check if the path exists
        is_frames = False
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The path '{input_path}' does not exist.")
        elif not os.path.isfile(input_path): # Detect if we are working with frames or a video
            logger.info(f"Input path is a directory. Loading frames from {input_path}")
            global_frames = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and (f.endswith('.jpg') or f.endswith('.png'))]
            global_frames = sorted(global_frames, key=lambda x: int(x.split('.')[0].split('_')[-1]))  # Sort the frames based on the frame number
            is_frames = True
        elif os.path.isfile(input_path): # We are working with a video
            video_extensions = {".mp4", ".avi", ".mov", ".mkv"}  # Add other common video extensions as needed
            file_extension = os.path.splitext(input_path)[-1].lower()
            if file_extension not in video_extensions:
                logger.error(f"Invalid video file format. Supported formats are {video_extensions}")
                raise Exception()
            else:
                logger.info(f"Input path is a video file. Loading video from {input_path}")
        else:
            logger.error(f"Invalid input path. Please provide a valid path to a video file or a directory containing frames.")
            raise Exception()

        if is_frames:
            # Load a single frame to get the video props
            with iio.imopen(os.path.join(input_path, global_frames[0]), io_mode="r") as file:
                frame = file.read()
            FRAME_HEIGHT, FRAME_WIDTH, _ = frame.shape
            total_n_frames = len(global_frames)
            video_FPS = 12.5 # Default FPS for frames
        else: # We are working with a video
            # Get the video props using OpenCV
            cap = cv2.VideoCapture(R"{}".format(input_path))
            if not cap.isOpened():
                logger.info(f"Error opening video file at {input_path}")
                raise Exception()
            else:
                logger.info(f"Successfully opened video file at {input_path}")

            total_n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get the total number of frames in the video
            video_FPS = cap.get(cv2.CAP_PROP_FPS)                   # Get the frame rate (FPS) of the video
            FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # Get the width of the frame
            FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the height of the frame

            # Load up the video using imageio
            reader = iio.imopen(input_path, io_mode='r', plugin='pyav').iter()

        # Get some Tracking Configs
        STATIONARY_THRESHOLD = tracking_cfg.STATIONARY_THRESHOLD    # Define a threshold for IoU below which we consider the vehicle as moving
        MIN_STATIONARY_FRAMES = tracking_cfg.MIN_STATIONARY_FRAMES  # Minimum number of frames a vehicle must be stationary to be considered stationary
        MIN_BOX_SIZE = tracking_cfg.MIN_BOX_SIZE                    # Minimum width or height of a bounding box to be considered complete
        TARGET_COLORS = misc_cfg.TARGET_COLOR                       # Define the target colors for the color model
        USE_FILTERS = misc_cfg.USE_FILTERS                          # Use filters to remove unwanted frames

        if USE_FILTERS:
            color_removed = 0           # Counter for number of frames removed due to color
            stationary_removed = 0      # Counter for number of frames removed due to stationary
            incomplete_removed = 0      # Counter for number of frames removed due to incomplete bounding boxes

        # Initialize dictionary to hold the previous frame's bounding boxes, classes, confidences, and cropped images
        chunk_dict = {}  # Temporary dictionary for chunk storage
        chunk_size = total_n_frames if total_n_frames < 3000 else 3000  # Define the number of frames per chunk

        # Initialize various counters counter
        total_frames_removed = 0    # Counter for total number of frames removed
        total_frames = 0            # Counter for total number of frames processed
        frame_number = 0            # Counter for the current frame number
        ids_removed = 0             # Counter for the number of IDs removed
        ids_to_remove = []          # List to store IDs that have no frames left

        # Load the mask if we are using it
        if (use_roi_mask == True):
            if (roi_mask_path != ""):
                mask = cv2.imread(roi_mask_path, 0)  # Grayscale mask
            else:
                # We need to extract a single frame from the video to create the mask
                mask = create_mask(frame, input_path)

            mask = cv2.resize(mask, (FRAME_WIDTH, FRAME_HEIGHT))
            logger.info(f"Successfully loaded ROI mask")

        # Initialize the progress bar with the total number of frames
        with tqdm(total=total_n_frames, desc="Processing Video", unit="frame") as pbar:

            # Run the tracking loop
            while frame_number < total_n_frames:

                # Retrieve frame based on input type
                if is_frames:
                    # Load the frame from the list of image files
                    with iio.imopen(os.path.join(input_path, global_frames[frame_number]), io_mode="r") as file:
                        frame_name = re.findall(r'\d+', global_frames[frame_number])[0]  # Extract the frame number from the filename
                        orig_frame = file.read()
                else:
                    # Load the next frame from the video
                    try:
                        frame_name = frame_number
                        orig_frame = next(reader)
                    except StopIteration:
                        break  # End of video stream
                
                frame_number += 1 # Increment frame counter

                # Calculate the timestamp for the current frame                
                timestamp = frame_number / video_FPS

                # Apply the mask: Keep only the region inside the mask (ROI)
                if (use_roi_mask == True):
                    roi_frame = cv2.bitwise_and(orig_frame, orig_frame, mask=mask)
                    frame = roi_frame
                    #cv2.imshow("ROI Frame", cv2.cvtColor(roi_frame, cv2.COLOR_RGB2BGR))
                else:
                    frame = orig_frame

                perf_timer.restart_timer()  # Restart the timer for the current frame

                # Run YOLO tracking on the frame
                results = yolo_model.track(frame, classes=list([idx for idx, _ in tracked_classes]), conf=conf_thresh,
                                            imgsz=yolo_img_size, iou=yolo_iou_threshold,
                                            agnostic_nms=use_agnostic_nms, tracker=tracker_yaml_file,
                                            device=device, persist=True, verbose=False)

                # Extract bounding boxes, classes, names, and confidences
                boxes = results[0].boxes.xyxy.tolist()
                classes = results[0].boxes.cls.tolist()
                ids = results[0].boxes.id.tolist() if results[0].boxes.id is not None else [-1] * len(classes)
                names = results[0].names
                confidences = results[0].boxes.conf.tolist()

                # Iterate over detected objects
                for box, cls, id, conf in zip(boxes, classes, ids, confidences):
                    if id == -1:
                        continue # Skip the object if the tracker does not detect it
                    else:
                        x1, y1, x2, y2 = box  # Convert to integers
                        # confidence = conf
                        detected_class = int(cls)
                        name = names[detected_class]
                        id = int(id)
                        crop = orig_frame[int(y1):int(y2), int(x1):int(x2)]  # Crop the bounding box from the image

                        # Add frame info to the chunk dictionary
                        if id not in chunk_dict:
                            chunk_dict[id] = []

                        chunk_dict[id].append({
                            'frame_number': frame_number,
                            'class': name,
                            'confidence': conf,
                            'bounding_box': box,
                            'cropped_image': crop,
                            'timestamp': round(timestamp, 2) # Round to 2 decimal places
                        })
                perf_timer.register_call("YOLO Object Detection + Tracking")

                # Flush and reset the dictionary after processing a chunk
                if frame_number % chunk_size == 0 or frame_number == total_n_frames:
                    total_frames += sum(len(v) for v in chunk_dict.values())  # Increment total frames counter

                    # Filter frames in chunk_dict before storing
                    if USE_FILTERS:
                        chunk_dict, frames_removed = filter_frames(chunk_dict, model, car_classifier,
                                                                   transforms, device, perf_timer,
                                                                    TARGET_COLORS, STATIONARY_THRESHOLD, MIN_STATIONARY_FRAMES,
                                                                    MIN_BOX_SIZE, FRAME_WIDTH, FRAME_HEIGHT)
                        total_frames_removed += sum(frames_removed.values())
                        color_removed += frames_removed['color']
                        stationary_removed += frames_removed['stationary']
                        incomplete_removed += frames_removed['incomplete']

                    # Before running the ReID model and returning the features, we must update the class of the bounding boxes
                    # We simply take the number of occurences of each class and assign the most frequent one to the bounding box
                    for id in chunk_dict:
                        frames = chunk_dict[id]
                        classes = [frame['class'] for frame in frames]
                        most_common_class = max(set(classes), key=classes.count)

                        # Remove all the classes occurrences from the frames, since we have a global variable for the class
                        for frame in frames: del frame['class']

                        chunk_dict[id] = {'class': most_common_class,
                                          'frames': [frame for frame in frames],}

                        # Sample frames from the trajectory to reduce the number of frames stored
                        if tracking_cfg.SAMPLE_FRAMES:
                            chunk_dict[id]['frames'] = sample_trajectory_frames(chunk_dict[id]['frames'], min_samples=10, max_samples=80)
                            
                            # Remove the ID if there are no frames left
                            if len(chunk_dict[id]['frames']) == 0:
                                logger.debug(f"ID {id} has no frames left, removing from the dictionary.")
                                ids_to_remove.append(id)
                                ids_removed += 1
                                continue
                        perf_timer.register_call("Minor filtering")
                    
                    # Update the chunk_dict with the filtered frames
                    if len(ids_to_remove) > 0:
                        for id in ids_to_remove:
                            del chunk_dict[id]
                        ids_to_remove = []  # Reset the list

                    # Run ReID model on the remaining frames
                    for id in tqdm(chunk_dict, desc="Running ReID model on remaining vehicles", unit="ID", total=len(chunk_dict)):
                        cropped_frames = [frame['cropped_image'] for frame in chunk_dict[id]['frames']]
                        for i, orig_frame in enumerate(cropped_frames):
                            frame = Image.fromarray(orig_frame).convert('RGB')  # Convert to PIL Image
                            frame = transforms.val_transform(frame)             # Convert to torch.Tensor
                            frame = frame.unsqueeze(0).float().to(device)

                            features = model.forward(frame, training=False)

                            # Add this feature to the dictionary for the ID
                            # id -> {class, frames: [{frame1}, {frame2}, ...]}
                            chunk_dict[id]['frames'][i]['features'] = features.detach().cpu().numpy()
                        perf_timer.register_call("Re-ID")

                    for id in tqdm(chunk_dict, desc="Trajectory creation", unit="ID", total=len(chunk_dict)):
                        # Combine Camera ID with Vehicle ID to create a unique ID for the vehicle
                        unique_vehicle_id = f"CAM{camera_data['_id']}_V{id}"

                        # Combine Vehicle ID with Trajectory ID
                        # Since we have one trajectory per vehicle, we can use the ID of the vehicle as the ID of the trajectory
                        # along with the suffix '_0' to indicate the first trajectory of the vehicle:
                        #  -> CAM0_V10_T0 : Trajectory 0 of Vehicle 10 from Camera 0
                        #  -> CAM0_V22_T0 : Trajectory 0 of Vehicle 22 from Camera 0
                        #  -> CAM1_V173_T0: Trajectory 0 of Vehicle 173 from Camera 1
                        trajectory_id = unique_vehicle_id + "_T0"
                    
                        chunk_dict[id]['_id'] = trajectory_id              # ID of the trajectory
                        chunk_dict[id]['vehicle_id'] = unique_vehicle_id   # ID of the vehicle
                        chunk_dict[id]['camera_id'] = camera_data['_id']   # ID of the camera

                        # Compress both the cropped imgs and the features into a suitable format for MongoDB
                        frames = chunk_dict[id]['frames']

                        for j, frame in enumerate(frames):
                            shape = frame['cropped_image'].shape
                            features = frame['features'].tolist()[0]
                            compressed_data = compress_img(frame['cropped_image']) # Compress the cropped image

                            # From now on, everything has been compressed, so insert it back to the Dict and start storing it
                            frame['compressed_image'] = compressed_data
                            frame['shape'] = shape
                            frame['features'] = features
                            frame['vehicle_id'] = unique_vehicle_id

                            # Create a unique ID for the bounding box
                            # Format: CAM0_V10_F0/1/2/3/... (Camera 0, Vehicle 10, Frame 0/1/2/3/...)
                            bbox_id = unique_vehicle_id + f"_F{j}"
                            if bbox_id in chunk_dict[id]['frames']:
                                logger.info(f"Frame with ID {bbox_id} already exists in the dictionary.")
                                # Take the last frame ID from the dictionary and increment it by 1
                                last_frame_id = chunk_dict[id]['frames'][-1]['_id']
                                last_frame = int(last_frame_id.split('_')[-1][1:])
                                bbox_id = unique_vehicle_id + f"_F{last_frame + 1}"

                            frame['_id'] = bbox_id
                                
                        # We need the start time and the ending time, which we can get from the timestamps
                        # of the first and last frames
                        start_time = frames[0]['timestamp']
                        end_time = frames[-1]['timestamp']

                        chunk_dict[id]['start_time'] = start_time
                        chunk_dict[id]['end_time'] = end_time
                        perf_timer.register_call("Trajectory creation")

                    if USE_DB:
                        logger.info("Inserting Vehicles, Trajectories & BBoxes into the database...")
                        for id in tqdm(chunk_dict, desc="Inserting Vehicles, Trajectories & BBoxes into the database...", unit="ID", total=len(chunk_dict)):     
                            # Compress both the cropped imgs and the features into a suitable format for MongoDB
                            frames = chunk_dict[id]['frames']
                            unique_vehicle_id = chunk_dict[id]['vehicle_id']
                            vehicle_class = chunk_dict[id]['class'] # Get the class of the vehicle

                            # Check if the vehicle already exists. This could happen if sometimes, the vehicle is
                            # being detected between two different chunks of frames.
                            # For example: In the first chunk, the vehicle is detected at frame 100, and in the second chunk,
                            # the vehicle is detected at frame 2151. In this case, that vehicle was being added to the database
                            # already at frame 100.
                            if not reid_DB.vehicles_col.find_one({'_id': unique_vehicle_id}):
                                reid_DB.insert_vehicle({'_id': unique_vehicle_id, 'class': vehicle_class})
                            else:
                                logger.info(f"Vehicle with ID {unique_vehicle_id} already exists, skipping insertion in Vehicles collection.")

                            for j, frame in enumerate(frames):
                                if reid_DB.bboxes_col.find_one({'_id': frame['_id']}):
                                    logger.info(f"Frame with ID {bbox_id} already exists in the Database.")
                                    # If the frame already exists, we should take the last frame and increment it by 1
                                    last_bbox_id = reid_DB.bboxes_col.aggregate([
                                        {'$match': {'vehicle_id': unique_vehicle_id}},
                                        {'$sort': {'_id': -1}},
                                        {'$limit': 1},
                                        {'$project': {'_id': 1}}
                                    ])._data[0]['_id']

                                    # Extract the last frame from the BBox ID
                                    last_frame = int(last_bbox_id.split('_')[-1][1:])

                                    frame['_id'] = unique_vehicle_id + f"_F{last_frame + 1}"

                                # Insert the bounding box into the database
                                # We do not insert the cropped_image, since we compressed the image for MongoDB purposes
                                reid_DB.insert_bbox({
                                    'frame_number': frame['frame_number'],
                                    '_id': frame['_id'],
                                    'vehicle_id': unique_vehicle_id,
                                    'compressed_image': frame['compressed_image'],
                                    'bounding_box': frame['bounding_box'],
                                    'confidence': frame['confidence'],
                                    #'cropped_image': frame['cropped_image'],
                                    'features': frame['features'],
                                    'shape': frame['shape'],
                                    'timestamp': frame['timestamp'],
                                })

                            # It means that the trajectory already exists in the database
                            # Hence, we should update the trajectory with new frames instead of inserting a new one
                            trajectory_id = chunk_dict[id]['_id']

                            if reid_DB.trajectories_col.find_one({'_id': trajectory_id}):
                                update = True
                            else:
                                update = False    

                            # Insert the data into the MongoDB database
                            doc = {
                                '_id': chunk_dict[id]['_id'],
                                'vehicle_id': chunk_dict[id]['vehicle_id'],
                                'camera_id': chunk_dict[id]['camera_id'],
                                'start_time': start_time,
                                'end_time': end_time,
                                #'trajectory_data': frames,
                            }

                            # Proceed with insertion only if size is acceptable
                            if (len(bson.BSON.encode(doc)) >= 16777216):  # 16MB limit for MongoDB
                                logger.error(f"Document size is too large for Trajectory {trajectory_id}. Skipping insertion of Vehicle {unique_vehicle_id}.")
                                ids_to_remove.append(id)
                                ids_removed += 1
                                continue
                            else:
                                reid_DB.insert_trajectory(doc, update=update)
                            perf_timer.register_call("Database")

                    # Update the chunk_dict with the filtered frames
                    if len(ids_to_remove) > 0:
                        for id in ids_to_remove:
                            del chunk_dict[id]
                        ids_to_remove = []  # Reset the list

                    # Iterate through each ID in chunk_dict
                    if tracking_cfg.SAVE_BOUNDING_BOXES:
                        for id in chunk_dict:
                            # Create a directory for the ID if it doesn't exist
                            folder_path = os.path.join(os.path.join(tracking_cfg.BOUNDING_BOXES_OUTPUT_PATH, camera_data['name'], f"{id}"))
                            if not os.path.exists(folder_path):
                                os.makedirs(folder_path)

                            # Extract the cropped images for each frame
                            frames = [frame['cropped_image'] for frame in chunk_dict[id]['frames']]

                            # Save each frame in the directory
                            for i, frame in enumerate(frames):
                                # Convert the numpy array (cropped_image) to a PIL Image
                                img = Image.fromarray(frame)
                                
                                # Take the last index of the frame and increment it by 1
                                file_names = os.listdir(folder_path)
                                if file_names:
                                    file_names = [file.split('.')[0].split('_')[1] for file in file_names if file.endswith('.jpg')]
                                    file_names = sorted([int(file) for file in file_names])
                                    index = file_names[-1] + 1 # Increment by 1 to get the next frame
                                else:
                                    index = i

                                img_save_path = os.path.join(folder_path, f"frame_{index}.jpg")

                                # Save the image
                                img.save(img_save_path)

                                #print(f"Saved frame {i} for ID {id} at {img_save_path}")

                        logger.info(f"Bounding boxes saved for chunk {frame_number // chunk_size}")

                    # Final insertion of the data in a .txt file
                    if cfg_file.TRACKING.GT:
                           for id in chunk_dict:
                            # Get all relevant information about this ID
                            data = chunk_dict[id]

                            frames = data['frames']

                            bboxes = [frame['bounding_box'] for frame in frames]
                            frame_numbers = [frame['frame_number'] for frame in frames]
                        
                            # Write formatted output to the file
                            # ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Conf', 'Xworld', 'Yworld', 'Zworld']
                            for bbox, frame_num in zip(bboxes, frame_numbers):
                                x1, y1, x2, y2 = bbox
                                width = int(round(x2 - x1))
                                height = int(round(y2 - y1))
                                res['frame'].append(frame_num)
                                res['track_id'].append(id)
                                res['bbox_topleft_x'].append(int(x1))
                                res['bbox_topleft_y'].append(int(y1))
                                res['bbox_width'].append(width)
                                res['bbox_height'].append(height)

                    # After the chunk is stored, Python's garbage collector will handle memory,
                    # but we can also explicitly free memory if necessary.
                    del chunk_dict

                    gc.collect()                # Run garbage collector
                    torch.cuda.empty_cache()    # If running on GPU, clear cache to free up memory

                    chunk_dict = {}             # Clear the dictionary to free memory

                # Update the progress bar after processing each frame
                pbar.update(1)

        # Release video capture and destroy all OpenCV windows
        if is_frames == False:
            cap.release()
        cv2.destroyAllWindows()

        print("===============================================")
        logger.info("Pipeline completed successfully! Here's a summary of the results:")
        logger.info(f"\t- Total BBoxes: {total_frames}")
        if USE_FILTERS:
            logger.info(f"\t- BBoxes removed: {total_frames_removed} ({(total_frames_removed / total_frames) * 100:.2f}%)")
            logger.info(f"\t\t- Frames removed due to color: {color_removed}")
            logger.info(f"\t\t- Frames removed due to stationary vehicles: {stationary_removed}")
            logger.info(f"\t\t- Frames removed due to incomplete bounding boxes: {incomplete_removed}")
        logger.info(f"\t- Total IDs: {len(set(res['track_id']))}")
        logger.info(f"\t- IDs removed due to errors: {ids_removed}")
        print("===============================================")

        if cfg_file.TRACKING.GT:
            # MOTChallenge format for Dataframe
            df = pd.DataFrame(res)
            df.columns   = ['frame', 'track_id', 'bbox_topleft_x', 'bbox_topleft_y', 'bbox_width', 'bbox_height']
            df['conf']   = 1
            df['Xworld'] = -1
            df['Yworld'] = -1
            df['Zworld'] = -1
            df.to_csv(output_file, index=False, header=False)

    if misc_cfg.RUN_UNIFICATION:

        # =============== UNIFY DIFFERENT IDs BOUNDING BOXES FROM SAME CAMERA ===============
        # After the pipeline is done, we must refine the BBoxes to be the same if
        # the similarity between features is above a certain threshold
        # and the temporal constraint are respected (due to Tracking mismatch).
        # In this case temporal constraint is given by the difference of start_time
        # and end_time of a trajectory.
        logger.info("Running unification across SINGLE camera...")

        # max_similarity_per_vehicle is a list of tuples containing the vehicle ID, the most similar vehicle ID, and the similarity
        max_similarity_per_vehicle = get_max_similarity_per_vehicle(reid_DB, reid_cfg, camera_data, device,
                                                                    similarity_algorithm, similarity_method)

        # Check if any vehicle similarity exceeds the threshold
        if not any(similarity >= similarity_threshold for _, _, similarity in max_similarity_per_vehicle):
            logger.warning("No similar vehicles detected across cameras. Skipping unification.")
            return

        # Create a graph where nodes are vehicle IDs
        G = GraphEngine()

        # Print out vehicle pairs with the highest similarity per vehicle
        for (veh1, veh2, similarity) in max_similarity_per_vehicle:
            G.add_node(veh1)  # Add the vehicle (bbox_id) as a node
            if similarity >= similarity_threshold:
                print(f"\tVehicle {veh1} is most similar to Vehicle {veh2} with similarity {similarity:.2f}")
                # In this case, we simply add an edge between the two vehicles, meaning that they are linked to each other.
                G.add_edge(veh1, veh2, weight=similarity)
            perf_timer.register_call("Unification")

        # Find all connected components in the graph (useful if we have multiple vehicles to merge)
        connected_components = G.get_connected_components()

        # Merge vehicles based on connected components
        for component in connected_components:
            component_list = list(component)
            if len(component_list) > 1:
                print(f"Trying to merge the following vehicles: {component_list}")
                component_list = sorted(component_list, key=extract_numeric_id)

                # We need to further filter the component list by the timestamp.
                # It means that we do need to enter each vehicle and check the timestamp of their trajectory
                # If the timestamp is close enough, we can consider them as the same vehicle
                # If the timestamp is too far apart, we can consider them as different vehicles
                # We can use the start_time and end_time of the trajectory to determine this

                traj_dict = {}

                for v in component_list:
                    # Trajectory ID:
                    #   - vehicle_id
                    #   - camera_id
                    #   - start_time
                    #   - end_time
                    #   - trajectory_data
                    trajectory = reid_DB.get_vehicle_single_trajectory(f'CAM{camera_data["_id"]}_V{v}')

                    if v not in traj_dict:
                        traj_dict[v] = {}

                    indices_traj = list(trajectory['trajectory_data'].keys())

                    traj_dict[v] = {
                        'start_time': trajectory['start_time'],
                        'end_time': trajectory['end_time'],
                        'bbox_start_trajectory': trajectory['trajectory_data'][indices_traj[0]][3],  # XYXY format
                        'bbox_end_trajectory': trajectory['trajectory_data'][indices_traj[-1]][3]    # XYXY format
                    }

                # Now add a value ['difference'] to the dictionary to store the difference between the start_time and end_time of i-1 and i indices
                # If the difference is greater than 1.50 seconds, it means they should be treated as different vehicles.
                # If the difference is less than 1.50 seconds, it means they should be treated as the same vehicle.

                traj_dict_copy = traj_dict.copy()
                for i, v in enumerate(traj_dict):
                    if i == 0:
                        continue

                    idx_vehicle_to_check = list(traj_dict.keys())[i - 1]
                    difference = abs(traj_dict[v]['start_time'] - traj_dict[idx_vehicle_to_check]['end_time'])
                    traj_dict[v]['difference'] = difference

                    # if difference is greater than 1.5 seconds, then remove the vehicle from the dictionary (it means that the vehicle is different)
                    if difference > 1.50:
                        traj_dict_copy.pop(v)
                    else:
                        # In this case, the difference is telling us that the vehicles could be the same
                        # hence, we check for the distance between the two vehicles
                        # If the distance is less than 1.5 meters, we consider them as the same vehicle
                        # If the distance is greater than 1.5 meters, we consider them as different vehicles

                        # Calculate centers for bounding boxes
                        def center_point(bbox):
                            return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                        center_bbox_start_i = center_point(traj_dict[v]['bbox_start_trajectory'])
                        center_bbox_end_i = center_point(traj_dict[v]['bbox_end_trajectory'])
                        center_bbox_start_j = center_point(traj_dict[idx_vehicle_to_check]['bbox_start_trajectory'])
                        center_bbox_end_j = center_point(traj_dict[idx_vehicle_to_check]['bbox_end_trajectory'])

                        # Calculate the distance between the two vehicles
                        distance_start = np.linalg.norm(np.array(center_bbox_start_i) - np.array(center_bbox_start_j))
                        # distance_end = np.linalg.norm(np.array(center_bbox_end_i) - np.array(center_bbox_end_j))

                        # If the distance is greater than 40 pixels, it means they should be treated as different vehicles
                        if distance_start > 40:
                            traj_dict_copy.pop(v)

                # Update the database by:
                # 1. Updating the bounding boxes
                # 2. Updating the trajectories
                # 3. Removing the vehicles from the database
                final_component_list = list(traj_dict_copy.keys())
                if len(final_component_list) > 1:
                    print(f"Merging vehicles: {final_component_list}")

                    # primary_vehicle = final_component_list[0]  # Choose the first vehicle as the primary one
                    # for other_vehicle in final_component_list[1:]:
                        # reid_DB.update_bbox(primary_vehicle, other_vehicle)
                        # reid_DB.update_trajectory(primary_vehicle, other_vehicle)
                        # reid_DB.remove_vehicle(other_vehicle)

                        # if tracking_cfg.SAVE_BOUNDING_BOXES:
                        #     # Move the bounding box images from the other vehicle to the primary vehicle
                        #     other_vehicle_folder = os.path.join(os.path.join(tracking_cfg.BOUNDING_BOXES_OUTPUT_PATH, camera_data['name'], f"{other_vehicle.split('V')[-1]}"))
                        #     primary_vehicle_folder = os.path.join(os.path.join(tracking_cfg.BOUNDING_BOXES_OUTPUT_PATH, camera_data['name'], f"{primary_vehicle.split('V')[-1]}"))
                            
                        #     # Get latest frame_number from the new vehicle_id and increment it by 1
                        #     frame_number = int((sorted([frame for frame in os.listdir(primary_vehicle_folder) if frame.endswith('.jpg')])[-1]).split('_')[-1].split('.')[0]) + 1

                        #     # Move all the files from the other vehicle folder to the primary vehicle folder
                        #     if os.path.exists(other_vehicle_folder):
                        #         if os.path.exists(primary_vehicle_folder):
                        #             for i, file in enumerate(os.listdir(other_vehicle_folder)):
                        #                 file_path = os.path.join(other_vehicle_folder, file)
                        #                 output_path = os.path.join(primary_vehicle_folder, f"frame_{frame_number + i}.jpg")
                        #                 shutil.move(file_path, output_path)
                        #         else:
                        #             shutil.move(other_vehicle_folder, primary_vehicle_folder)
                        #         shutil.rmtree(other_vehicle_folder)
            perf_timer.register_call("Unification")
        print(f"Single camera unification complete for Camera with ID: {camera_data['_id']}.")
        # =================================================================================== #
    logger.info(f"MOT Benchmark (times in ms):\n{perf_timer.get_benchmark()}")

# === WIP! === #
def run_mtmc(similarity_algorithm, similarity_method, similarity_method_query):
    """Cross-camera analysis to link vehicles across multiple cameras."""
    print("Running cross-camera analysis for MTMC...")

    # Logger
    logger = Logger()

    # DB connection
    USE_DB = db_cfg.USE_DB
    reid_DB = Database(db_configs=db_cfg, logger=logger) if USE_DB else None

    # =============== UNIFY DIFFERENT IDs BOUNDING BOXES FROM DIFFERENT CAMERAs =============== #
    # Gather all frames from the Database, separated by vehicle ID
    # For each vehicle ID, we will have a list of frames
    # We will then run the ReID model on each frame and compare the features
    # If the features are similar, we will consider them as the same vehicle
    # If the features are different, we will consider them as different vehicles
    all_frames = reid_DB.get_all_frames()

    # We must extract a query image from each vehicle and compare it with the gallery images of trajectories of different cameras
    # We will then compute the similarity between the query image and the gallery images
    # If the similarity is high, we will consider them as the same vehicle
    # If the similarity is low, we will consider them as different vehicles
    # Take the total number of vehicles in all_frames
    total_num_frames = 0
    col_names, row_names = [], []                

    # Iterate through each camera
    for cam in all_frames:
        # Iterate through each vehicle in the camera
        for vehicle in all_frames[cam]:
            # Query Image. We'll take the middle frame of the trajectory
            total_num_frames += len(all_frames[cam][vehicle])
            idx = int(len(all_frames[cam][vehicle]) / 2)

            # Iterate through each frame of the vehicle
            for i, frame_id in enumerate(all_frames[cam][vehicle]):
                # Create column names for each frame
                if i == idx:
                    row_names.append(f"Q_{frame_id}")
                else:
                    col_names.append(f"G_{frame_id}")

    # Create a matrix of similarities
    sim_matrix = np.zeros((len(row_names), len(col_names)))

    # Initialize a pandas DataFrame to store the matrix of similarities
    df = pd.DataFrame(sim_matrix, index=row_names, columns=col_names)

    # We need to fill the df now in this way:
    # - Iterate over the DataFrame rows (which are the queries)
    # - Take the feature of that query
    # - Iterate over each row, making sure to skip the gallery images coming from the same car and the gallery images coming from the same camera. Everything else should be computed with a similarity measure.
    with tqdm(total=len(row_names) * len(col_names), desc="Computing Similarity Matrix") as pbar:
        for query_idx, query_name in enumerate(row_names):
            # Extract camera and vehicle information from the query row name
            query_cam, query_vehicle, query_frame = query_name.split('_')[1:4] # ['CAM0', 'V20', 'F17']
            query_cam_id = int(query_cam.split("CAM")[-1])  # Camera number extraction | 'CAM0' -> 0
            query_name = query_name.split('Q_')[-1]  # Extract the frame number | 'Q_CAM0_V20_F17' -> 'CAM0_V20_F17'
            
            # Retrieve the feature for the query based on the similarity method
            if similarity_method_query == 'mean':
                frames = all_frames[query_cam_id][f'{query_cam}_{query_vehicle}']
                query_feature = np.mean([frame_data[0] for frame_data in frames.values()], axis=0)
            elif similarity_method_query == 'individual':
                query_feature = all_frames[query_cam_id][f'{query_cam}_{query_vehicle}'][f'{query_name}'][0]
            else:
                raise ValueError(f"Invalid similarity method: {similarity_method_query}")

            # Loop through each gallery image in the DataFrame (columns)
            for gallery_idx, gallery_name in enumerate(col_names):
                # Extract camera and vehicle from the gallery column name
                gallery_cam, gallery_vehicle, gallery_frame = gallery_name.split('_')[1:4]
                
                # Skip if the gallery image is from the same camera
                if query_cam == gallery_cam:
                    pbar.update(1)
                    continue

                gallery_cam_id = int(gallery_cam.split("CAM")[-1])  # Camera number extraction
                gallery_name = gallery_name.split('G_')[-1]  # Extract the frame number | 'G_CAM0_V20_F17' -> 'CAM0_V20_F17'
                gallery_feature = all_frames[gallery_cam_id][f'{gallery_cam}_{gallery_vehicle}'][f'{gallery_name}'][0]

                # Compute the similarity between the query and gallery features
                if similarity_algorithm == 'euclidean':
                    similarity = np.linalg.norm(query_feature - gallery_feature)
                elif similarity_algorithm == 'cosine':
                    similarity = cosine_similarity_np(query_feature, gallery_feature)
                else:
                    raise ValueError(f"Invalid similarity algorithm: {similarity_algorithm}")

                # Store the result in the DataFrame
                df.iloc[query_idx, gallery_idx] = similarity
                pbar.update(1)

    # ======= !!! IMPORTANT !!! ======= #
    # Before showing the predictions, we might need to filter again for the Top-K predictions based on the timestamp and direction
    # This is because we might have very very similar cars (even duplicates sometimes - That's the case of a GTA V dataset)
    # ================================= #

    # Remove trajectories below the threshold
    df_cleaned, traj_removed = remove_below_threshold(df, threshold=removing_threshold, method=similarity_method)

    # Compare trajectories between cameras
    # Threshold here is for the mean of the gallery features compared with the query feature
    possible_traj, G = compare_trajectories(reid_DB, df_cleaned, threshold=comparing_threshold, method=similarity_method)
    
    connected_components = G.get_connected_components() # Find all connected components in the graph

    # Sort, for each key, the values by the similarity
    for key in possible_traj:
        possible_traj[key] = sorted(possible_traj[key], key=lambda x: x[list(x.keys())[0]][0], reverse=True)
    # ================================= #

    # # Transform data into a structured format for DataFrame
    # rows = []
    # columns = []
    # values = []

    # # Extract data into lists for DataFrame creation
    # for (query, _), galleries in possible_traj.items():
    #     for gallery in galleries:
    #         for gallery_id, (similarity, _) in gallery.items():
    #             rows.append(query)
    #             columns.append(gallery_id)
    #             values.append(similarity)

    # # Create a DataFrame with the similarity matrix
    # similarity_df = pd.DataFrame({'Query': rows, 'Gallery': columns, 'Similarity': values})
    # similarity_matrix = similarity_df.pivot(index="Query", columns="Gallery", values="Similarity")

    # # Plot using seaborn heatmap
    # import seaborn as sns
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(similarity_matrix, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={'label': 'Similarity Score'})
    # plt.title("Trajectory Similarity Matrix")
    # plt.xlabel("Gallery Trajectories")
    # plt.ylabel("Query Trajectories")
    # plt.show()

    K = 3 # Top K predictions to show
    vehicles_to_show = 8 # len(list(possible_traj.keys()))/ integer_number (e.g. 4)
    possible_traj = dict(list(possible_traj.items())[-vehicles_to_show:])
    # possible_traj = dict(list(possible_traj.items())[:vehicles_to_show])

    fig, axes = plt.subplots(vehicles_to_show, K+1, figsize=(12, 15))
    query_trans = T.Pad(4, 0)
    good_trans = T.Pad(4, (0, 255, 0))          # Green
    bad_trans = T.Pad(4, (255, 0, 0))           # Red
    empty_trans = T.Pad(4, (255, 255, 0, 255))  # Yellow

    # Create a transparent image with RGBA mode
    width, height = 128, 128  # Specify your desired dimensions
    transparent_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))  # (0, 0, 0, 0) is fully transparent
    transparent_img = empty_trans(transparent_img)

    # Set column titles
    axes[0, 0].set_title("Query")
    for j in range(1, K+1):
        axes[0, j].set_title(f"Prediction #{j}")

    for i, ((query, _), galleries) in enumerate(possible_traj.items()):
        if i == vehicles_to_show: break

        # Safely remove all axis from the plot
        for ax in axes[i]:
            ax.axis("off")

        q_cam = query.split('_')[1]                 # 'CAM0'
        q_cam_id = int(q_cam.split('CAM')[-1])      # 0
        q_vid = q_cam + '_' + query.split('_')[2]   # 'CAM0_V10'
        query_img = load_middle_frame(tracking_cfg.BOUNDING_BOXES_OUTPUT_PATH, q_vid, q_cam_id)
        query_img = query_trans(query_img)
        ax = axes[i, 0]  # Display in the first column of the row
        ax.imshow(query_img)

        # Special case for when a query does not have any predictions
        if len(galleries) == 0:
            for j in range(0, K):
                ax = axes[i, j+1]
                ax.imshow(transparent_img)
            continue
                
        # Plot the top K predictions
        # for j in range(min(K, len(galleries))):
        for j in range(0, K):
            ax = axes[i, j+1]  # Display in subsequent columns
            try:
                gvid, (sim, _) = list(galleries[j].items())[0]
            except:
                ax.imshow(transparent_img)
                continue
            g_cam = gvid.split('_')[0]              # 'CAM1'
            g_cam_id = int(g_cam.split('CAM')[-1])  # 1
            img_pred = load_middle_frame(tracking_cfg.BOUNDING_BOXES_OUTPUT_PATH, gvid, g_cam_id)
            if j == 0:
                img_pred = good_trans(img_pred)
            else:
                img_pred = bad_trans(img_pred)

            ax.title.set_text(f"{sim:.2f}")
            ax.imshow(img_pred)

    plt.tight_layout()
    plt.show()
    plt.pause(200000) # Pause for the Debug mode

    # # ========================================================================================= #