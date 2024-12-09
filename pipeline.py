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
# - Update trajectories, bounding boxes, and vehicles in the database           -> Done
# - Link trajectories together and run MOTA metrics                             -> Done
# - Create Target Search Engine                                                 -> Done
# - Create a GUI for the pipeline                                               -> 0%

import gc
import os
import pickle
import random
import re
import shutil
import time
from datetime import datetime

import bson
import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from torchvision import transforms as T
from tqdm import tqdm

from config import _C as cfg_file
from db.database import Database
from misc.printer import Logger
from misc.timer import PerformanceTimer
from misc.utils import (compress_img, create_mask, decompress_img,
                        sample_trajectory_frames, set_seed)
from reid.datasets.transforms import Transformations
from reid.model import ModelBuilder
from tracking.cameras import CameraLayout
from tracking.filter import filter_frames
from tracking.model import load_yolo
from tracking.track import Track, Trajectory
from tracking.unification import get_max_similarity_per_vehicle


def fliplr(img):
    """ flip a batch of images horizontally """
    # N x C x H x W
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extract_image_patch(image, bbox, patch_shape=None):
    """Extract an image patch (defined by a bounding box) from an image
    Parameters
    ----------
    image : torch.tensor | ndarray
        The full image (3 dimensions, with an RGB or grayscale channel).
    bbox : array_like
        The bounding box in format (tx, ty, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.
    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.
    """

    if len(image.shape) != 3:
        raise NotImplementedError(
            "Only image arrays of 3 dimensions are supported.")

    if image.shape[0] <= 3:
        ch, h, w = image.shape
    elif image.shape[2] <= 3:
        h, w, ch = image.shape
    else:
        raise ValueError(
            "Input image does not contain an RGB or gray channel.")

    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int32)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum([w, h], bbox[2:])

    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox

    if image.shape[0] <= 3:
        image = image[:, sy:ey, sx:ex]
    else:
        image = image[sy:ey, sx:ex, :]
    return image

# Various configs
augmentation_cfg = cfg_file.REID.AUGMENTATION
misc_cfg = cfg_file.MISC
reid_cfg = cfg_file.REID
tracking_cfg = cfg_file.TRACKING
tracker_cfg = cfg_file.TRACKER
yolo_cfg = cfg_file.YOLO
detector_cfg = cfg_file.DETECTOR
db_cfg = cfg_file.DB

# Number of classes in each dataset (Only for ID classification tasks, hence only training set is considered)
num_classes = {
    'ai_city': 440,
    'ai_city_sim': 1802,
    'vehicle_id': 13164,
    'veri_776': 576,
    'veri_wild': 30671,
    'vric': 2811,
    'vru': 7086,
}

# SIMILARITY_THRESHOLD
similarity_threshold = 0.60
removing_threshold = 0.60
comparing_threshold = 0.60

def any_compatible(mtrack1, mtrack2, cams) -> bool:
    """Is there a pair of tracks between mtrack1 and mtrack2 that are compatible?"""
    for track1 in mtrack1.tracks:
        for track2 in mtrack2.tracks:
            if tracks_compatible(track1, track2, cams):
                return True
    return False

def have_mutual_cams(mtrack1, mtrack2) -> bool:
    """Checks whether two mutlicam tracklets share any cameras."""
    return bool(mtrack1.cams & mtrack2.cams)

def tracks_compatible(track1, track2, cams) -> bool:
    """Check whether two tracks can be connected to each other."""
    cam1, cam2 = track1.cam, track2.cam

    # if there is no cam layout, we only check if the tracks are on the same camera
    if cams is None:
        return cam1 != cam2
    
    # same camera -> they cannot be connected
    if cam1 == cam2:
        return False

    t1_start, t1_end = track1.start_time, track1.end_time
    t2_start, t2_end = track2.start_time, track2.end_time

    # is track1 -> track2 transition possible?
    # for this, [t2_start, t2_end] has to intersect with interval I = [t1_end + dtmin, t1_end + dtmax]
    # that is: track2 starts before I ends and I starts before track2 ends
    if (cams.cam_compatibility_bitmap(cam1) & (1 << cam2) > 0) and \
       t2_start <= t1_end + cams.dtmax[cam1][cam2] and \
       t1_end + cams.dtmin[cam1][cam2] <= t2_end:
        return True

    # check the track2 -> track1 transition too
    if (cams.cam_compatibility_bitmap(cam2) & (1 << cam1) > 0) and \
       t1_start <= t2_end + cams.dtmax[cam2][cam1] and \
       t2_end + cams.dtmin[cam2][cam1] <= t1_end:
        return True
    return False

# These are MultiTracks
def compute_similarity(track1, track2, linkage, already_normed, df):
    if linkage == "mean_feature":
        f1 = track1.mean_feature
        f2 = track2.mean_feature
        if already_normed:
            return np.dot(f1, f2)
        else:
            return np.dot(f1, f2) / np.linalg.norm(f1, 2) / np.linalg.norm(f2, 2)
    
    # We must retrieve the similarities between all pairs of frames in the DataFrames
    all_sims = []
    for t1 in track1.tracks:
        for t2 in track2.tracks:
            all_sims.append(df.iloc[t1.id, t2.id])

    if linkage == "average":
        return np.mean(all_sims)
    if linkage == "single":
        return np.max(all_sims)
    if linkage == "complete":
        return np.min(all_sims)

def load_models_and_transformations(device, cfg):
    """Load YOLO, ReID model, color model, and transformations."""
    # Load YOLO model
    yolo_model = load_yolo(cfg)
    yolo_model = yolo_model.to(device)

    # Load ReID Model
    dataset_name = reid_cfg.DATASET.DATASET_NAME

    # ================================================================
    cfg.REID.MODEL.PADDING_MODE = cfg.REID.AUGMENTATION.PADDING_MODE
    reid_model = ModelBuilder(
        model_name=cfg.REID.MODEL.NAME,
        pretrained=cfg.REID.MODEL.PRETRAINED,
        num_classes=num_classes[dataset_name],
        model_configs=cfg.REID.MODEL,
        device=device
    ).move_to(device)

    # Load parameters from a .pth file
    val_path = cfg.REID.TEST.MODEL_VAL_PATH
    if (val_path is not None):
        print("Loading model parameters from file...")
        if ('ibn' in cfg.REID.MODEL.NAME):
            reid_model.load_param(val_path)
        else:
            checkpoint = torch.load(val_path, map_location=device)
            reid_model.load_state_dict(checkpoint['model'])
        print(f"Successfully loaded model parameters from file: {val_path}")

    reid_model.eval() # Set the model to evaluation mode

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
    return yolo_model, reid_model, car_classifier, transforms

def load_config_and_device(config_path=None):
    """Load config, initialize device and set seed."""
    if config_path:
        cfg_file.merge_from_file(config_path)
    device = cfg_file.MISC.DEVICE
    if device == 'cuda' and torch.cuda.is_available():
        device = f"{device}:{cfg_file.MISC.GPU_ID}"
    else:
        device = 'cpu'
    return cfg_file, device

def setup_database():
    """Initialize the database connection."""
    logger = Logger()
    db = None
    if cfg_file.DB.USE_DB:
        db = Database(db_configs=cfg_file.DB, logger=logger)

        if cfg_file.DB.CLEAN_DB:
            logger.debug("Cleaning the database...")
            db.clean()
    return db

def run_mtsc(yolo_model, model, car_classifier, transforms, device, camera_data, db, logger):
    """Run the pipeline for a single camera, used in both MTSC and MTMC."""
    # Set seed
    set_seed(cfg_file.MISC.SEED)

    # Load PerformanceTimer
    perf_timer = PerformanceTimer()

    # DB connection
    USE_DB = db_cfg.USE_DB
    reid_DB = db if USE_DB else None

    # Get the similarity method to use
    similarity_algorithm    = reid_cfg.TEST.SIMILARITY_ALGORITHM    # 'cosine' / 'euclidean'
    unification_method      = reid_cfg.TEST.UNIFICATION_METHOD      # 'mean' / 'max'

    # If GT is available, we must run MOTA metrics
    cfg_file.TRACKING.GT = camera_data.get('gt', None)
    if cfg_file.TRACKING.GT:
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = f'MTSC-predictions_camera-{camera_data["_id"]}_{date}.txt'

        pickle_global_path = cfg_file.MTMC.PICKLE_GLOBAL_PATH
        if not os.path.exists(pickle_global_path):
            os.makedirs(pickle_global_path, exist_ok=True)

    if misc_cfg.RUN_PIPELINE:
        # Configure camera-specific parameters
        tracking_cfg.VIDEO_PATH     = camera_data['path']
        tracking_cfg.ROI_MASK_PATH  = camera_data.get('roi', None)

        # =========================== YOLO =========================== #
        tracked_classes     = yolo_cfg.TRACKED_CLASS
        conf_thresh         = yolo_cfg.CONFIDENCE_THRESHOLD
        yolo_img_size       = yolo_cfg.YOLO_IMAGE_SIZE
        yolo_iou_threshold  = yolo_cfg.YOLO_IOU_THRESHOLD
        use_agnostic_nms    = yolo_cfg.USE_AGNOSTIC_NMS
        verbose             = yolo_cfg.VERBOSE
        # ========================== Tracker ========================= #
        tracker_model       = tracker_cfg.MODEL.NAME
        tracker_yaml_file   = tracker_cfg.YOLO_TRACKER
        persist             = tracker_cfg.PERSIST
        # =========================== Misc ============================ #
        input_path          = tracking_cfg.VIDEO_PATH
        use_roi_mask        = tracking_cfg.USE_ROI_MASK
        roi_mask_path       = tracking_cfg.ROI_MASK_PATH
        # ============================================================ #
    
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
                                            device=device, persist=persist, verbose=verbose)

                # Extract bounding boxes, classes, names, and confidences
                boxes = results[0].boxes.xywh.tolist() # XYWH = CenterX, CenterY, Width, Height
                classes = results[0].boxes.cls.tolist()
                ids = results[0].boxes.id.tolist() if results[0].boxes.id is not None else [-1] * len(classes)
                names = results[0].names
                confidences = results[0].boxes.conf.tolist()

                # Iterate over detected objects
                for box, cls, id, conf in zip(boxes, classes, ids, confidences):
                    if id == -1:
                        continue # Skip the object if the tracker does not detect it
                    else:
                        x, y, w, h = map(int, box) # Convert the bounding box to integers

                        # Convert the bounding box to the tlwh format
                        #  => (top left x, top left y, width, height)
                        tlwh_box = [int(round(x - w / 2)), int(round(y - h / 2)), w, h]
                        tx, ty, tw, th = tlwh_box

                        # Crop the bounding box from the image
                        crop = orig_frame[ty:ty + th,tx:tx + tw]

                        # confidence = conf
                        detected_class = int(cls)
                        name = names[detected_class]
                        id = int(id)

                        # Add frame info to the chunk dictionary
                        if id not in chunk_dict:
                            chunk_dict[id] = []

                        chunk_dict[id].append({
                            'frame_number': frame_number,
                            'class': name,
                            'confidence': conf,
                            'bounding_box': tlwh_box, # tlwh format
                            'cropped_image': crop,
                            'frame': frame,
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
                        if tracking_cfg.DELETE_MIN_FRAMES:
                            # If the number of frames is less than the minimum, return an empty list
                            if len(chunk_dict[id]['frames']) < 5:
                                chunk_dict[id]['frames'] = []
                                #logger.debug(f"ID {id} has no frames left, removing from the dictionary.")
                                ids_to_remove.append(id)
                                ids_removed += 1
                                continue

                        if tracking_cfg.SAMPLE_FRAMES:
                            chunk_dict[id]['frames'] = sample_trajectory_frames(chunk_dict[id]['frames'], min_samples=5, max_samples=3000)
                            
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
                            del chunk_dict[id]['frames'][i]['frame']

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

                        # We need the start time and the ending time, which we can get
                        # from the timestamps of the first and last frames
                        start_time = frames[0]['timestamp']
                        end_time = frames[-1]['timestamp']

                        chunk_dict[id]['start_time'] = start_time
                        chunk_dict[id]['end_time'] = end_time
                        perf_timer.register_call("Trajectory creation")

                    if USE_DB:
                        # logger.info("Inserting Vehicles, Trajectories & BBoxes into the database...")
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
                                    'bounding_box': frame['bounding_box'], # tlwh format
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
                                'start_time': chunk_dict[id]['start_time'],
                                'end_time': chunk_dict[id]['end_time'],
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

        logger.info("===============================================")
        logger.info("Pipeline completed successfully! Here's a summary of the results:")
        logger.info(f"\t- Total BBoxes: {total_frames}")
        if USE_FILTERS:
            logger.info(f"\t- BBoxes removed: {total_frames_removed} ({(total_frames_removed / total_frames) * 100:.2f}%)")
            logger.info(f"\t\t- Frames removed due to color: {color_removed}")
            logger.info(f"\t\t- Frames removed due to stationary vehicles: {stationary_removed}")
            logger.info(f"\t\t- Frames removed due to incomplete bounding boxes: {incomplete_removed}")
        #logger.info(f"\t- Total IDs: {len(set(res['track_id']))}")
        logger.info(f"\t- IDs removed due to errors: {ids_removed}")
        logger.info("===============================================")

    if misc_cfg.RUN_UNIFICATION:

        # =============== UNIFY DIFFERENT IDs BOUNDING BOXES FROM SAME CAMERA ===============
        # After the pipeline is done, we must refine the BBoxes to be the same if:
        # 
        logger.info("Running unification across SINGLE camera...")

        # max_similarity_per_vehicle is a list of tuples containing the vehicle ID, the most similar vehicle ID, and the similarity
        max_similarity_per_vehicle = get_max_similarity_per_vehicle(reid_DB, reid_cfg, camera_data, device,
                                                                    similarity_algorithm, unification_method)

        # If max_similarity_per_vehicle is empty, it means that no similar vehicles were detected across cameras
        if not max_similarity_per_vehicle:
            return

        # Check if any vehicle similarity exceeds the threshold
        if not any(similarity >= similarity_threshold for _, _, similarity in max_similarity_per_vehicle):
            logger.warning("No similar vehicles detected across cameras. Skipping unification.")
        else:
            logger.info("Similar vehicles detected across cameras. Starting unification...")

            # Step 1: Define Thresholds for Merging
            TIME_GAP_THRESHOLD = 1.50           # Maximum allowed time gap in seconds
            SPATIAL_DISTANCE_THRESHOLD = 40.0   # Maximum allowed spatial distance in pixels

            # Function to compute spatial distance between bounding boxes
            def compute_spatial_distance(box1, box2):
                # Compute centers of the boxes (tlwh format)
                x1_center = box1[0] + box1[2] / 2
                y1_center = box1[1] + box1[3] / 2
                x2_center = box2[0] + box2[2] / 2
                y2_center = box2[1] + box2[3] / 2

                # Compute Euclidean distance between the centers
                distance = np.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)
                return distance

            # Step 4: Merge Tracklets Based on Similarity and Spatial-Temporal Proximity
            id_mapping = {}  # Mapping from old IDs to new IDs

            for tuple in max_similarity_per_vehicle:
                id_i, id_j, similarity = tuple[0], tuple[1], tuple[2]

                # Swap IDs if i > j (Meaning that we want the i to be the earlier car and j to be the later car)
                if id_i > id_j:
                    id_i, id_j = id_j, id_i

                # Skip if either tracklet has been merged into another
                root_id_i = id_mapping.get(id_i, id_i)
                root_id_j = id_mapping.get(id_j, id_j)

                if root_id_i == root_id_j:
                    continue

                if id_mapping.get(id_i) is not None or id_mapping.get(id_j) is not None:
                    continue # Skip if the ID has already been merged

                trajectory_i = reid_DB.get_vehicle_single_trajectory(f'CAM{camera_data["_id"]}_V{id_i}')
                trajectory_j = reid_DB.get_vehicle_single_trajectory(f'CAM{camera_data["_id"]}_V{id_j}')

                # Get the last detection of tracklet i and first detection of tracklet j
                # 'trajectory_data':
                #   - [0]: Features
                #   - [1]: Compressed Image
                #   - [2]: Shape
                #   - [3]: Bounding Box
                #   - [4]: Timestamp
                end_det_i = trajectory_i['trajectory_data'][list(trajectory_i['trajectory_data'].keys())[-1]]
                start_det_j = trajectory_j['trajectory_data'][list(trajectory_j['trajectory_data'].keys())[0]]

                # Compute time gap
                time_gap = start_det_j[4] - end_det_i[4]
                if time_gap < 0 or time_gap > TIME_GAP_THRESHOLD:
                    continue

                # Compute spatial distance
                spatial_distance = compute_spatial_distance(end_det_i[3], start_det_j[3])
                if spatial_distance > SPATIAL_DISTANCE_THRESHOLD:
                    continue

                # Merge tracklets
                logger.debug(f"Merging Tracklet {root_id_j} in Tracklet {root_id_i}")

                # Update id_mapping
                id_mapping[root_id_j] = root_id_i

                # Retrieve the DB IDs for the vehicles
                DB_id_i = f'CAM{camera_data["_id"]}_V{root_id_i}'
                DB_id_j = f'CAM{camera_data["_id"]}_V{root_id_j}'

                # Update the bounding boxes and trajectories
                reid_DB.update_bbox(DB_id_i, DB_id_j)
                reid_DB.update_trajectory(DB_id_i, DB_id_j)
                reid_DB.remove_vehicle(DB_id_j)

                if tracking_cfg.SAVE_BOUNDING_BOXES:
                    # Move the bounding box images from the other vehicle to the primary vehicle
                    primary_vehicle_folder = os.path.join(os.path.join(tracking_cfg.BOUNDING_BOXES_OUTPUT_PATH, camera_data['name'], f"{DB_id_i.split('V')[-1]}"))
                    other_vehicle_folder = os.path.join(os.path.join(tracking_cfg.BOUNDING_BOXES_OUTPUT_PATH, camera_data['name'], f"{DB_id_j.split('V')[-1]}"))
                    
                    # Get latest frame_number from the new vehicle_id and increment it by 1
                    frame_number = int((sorted([frame for frame in os.listdir(primary_vehicle_folder) if frame.endswith('.jpg')])[-1]).split('_')[-1].split('.')[0]) + 1

                    # Move all the files from the other vehicle folder to the primary vehicle folder
                    if os.path.exists(other_vehicle_folder):
                        if os.path.exists(primary_vehicle_folder):
                            for i, file in enumerate(os.listdir(other_vehicle_folder)):
                                file_path = os.path.join(other_vehicle_folder, file)
                                output_path = os.path.join(primary_vehicle_folder, f"frame_{frame_number + i}.jpg")
                                shutil.move(file_path, output_path)
                        else:
                            shutil.move(other_vehicle_folder, primary_vehicle_folder)
                        shutil.rmtree(other_vehicle_folder)

                perf_timer.register_call("Unification")
            logger.debug(f"Single camera unification complete for Camera with ID: {camera_data['_id']}.")
        # =================================================================================== #
    
    # Get the benchmark data
    logger.info("MTSC Benchmark (times in ms):")
    records = perf_timer.get_benchmark(unit="ms", precision=4, logger=logger)

    # Final insertion of the data in a .txt file
    if cfg_file.TRACKING.GT:
        logger.debug(f"Saving MTSC data for Camera {camera_data['_id']} to .txt file and pickle...")
        pickle_tracklets = []
        res = {
            "frame": [],
            "track_id": [],
            "bbox_topleft_x": [],
            "bbox_topleft_y": [],
            "bbox_width": [],
            "bbox_height": [],
        }

        # data['features'], data['compressed_image'], data['shape'],
        # data['frame_number'], data['bounding_box'], data['timestamp'],
        # data['confidence'], data['vehicle_id']
        all_frames = reid_DB.get_camera_frames(camera_data['_id'])
        for id in all_frames:
            # Get all relevant information about this ID
            frames = all_frames[id]

            # Get vehicle ID
            id = id.split('V')[-1]

            # Get properties
            features = [v[0] for _, v in frames.items()]
            # compressed_images = [v[1] for _, v in frames.items()]
            shapes = [v[2] for _, v in frames.items()]
            frame_numbers = [v[3] for _, v in frames.items()]
            bboxes = [v[4] for _, v in frames.items()]
            timestamps = [v[5] for _, v in frames.items()]
            confidences = [v[6] for _, v in frames.items()]
            vehicle_ids = [v[7] for _, v in frames.items()]

            # Write formatted output to the file
            # ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Conf', 'Xworld', 'Yworld', 'Zworld']
            for bbox, frame_num in zip(bboxes, frame_numbers):
                tx, ty, w, h = bbox
                res['frame'].append(frame_num)
                res['track_id'].append(id)
                res['bbox_topleft_x'].append(int(tx))
                res['bbox_topleft_y'].append(int(ty))
                res['bbox_width'].append(int(round(w)))
                res['bbox_height'].append(int(round(h)))

            # Append to pickle
            pickle_tracklets.append({
                'id': id,
                'features': features,
                'shapes': shapes,
                'frame_numbers': frame_numbers,
                'bboxes': bboxes,
                'timestamps': timestamps,
                'confidences': confidences,
                'vehicle_ids': vehicle_ids,
            })

        # MOTChallenge format for Dataframe
        df = pd.DataFrame(res)
        df.columns   = ['frame', 'track_id', 'bbox_topleft_x', 'bbox_topleft_y', 'bbox_width', 'bbox_height']
        df['conf']   = 1
        df['Xworld'] = -1
        df['Yworld'] = -1
        df['Zworld'] = -1
        df.to_csv(output_file, index=False, header=False)

        # Export the pickle file
        pickle_path = os.path.join(pickle_global_path, f"tracklets_cam{camera_data['_id']}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_tracklets, f)

    return records

def run_mtmc(similarity_method, db, cam_layout, cfg, logger):
    """Cross-camera analysis to link vehicles across multiple cameras."""

    # Camera Layout
    cams = CameraLayout(cam_layout['path'])

    # Set seed
    set_seed(cfg_file.MISC.SEED)
    logger.debug(f"Correctly set the seed to: {cfg_file.MISC.SEED}")

    logger.info("Running cross-camera analysis for MTMC...")

    # DB connection
    USE_DB = db_cfg.USE_DB
    reid_DB = db if USE_DB else None

    logger.info("Gathering all frames...")
    log_start_time = time.time()

    # Fallback to Pickle first
    if cfg_file.MTMC.USE_PICKLE:
        pickle_objs_files = os.listdir(cfg_file.MTMC.PICKLE_GLOBAL_PATH)

        # Filter only for files which ends with .pkl and starts with "tracklets_cam"
        pickle_objs_files = [file for file in pickle_objs_files if file.endswith('.pkl') and file.startswith('tracklets_cam')]

        if len(pickle_objs_files) != cams.n_cams:
            logger.error("Number of Pickle files does not match the number of cameras.")
            raise Exception()

        all_frames = {}
        for i, pickle_path in enumerate(pickle_objs_files):   # PICKLE SINGLE FILES
            path = os.path.join(cfg_file.MTMC.PICKLE_GLOBAL_PATH, pickle_path)
            with open(path, "rb") as f:
                res = pickle.load(f)
                # We must have the same schema as get_all_frames() from the Database
                for r in res:
                    vehicle_id = r['vehicle_ids'][0]  # Assuming vehicle_id is consistent across the frames

                    # Loop over frame numbers and other associated data
                    for j in range(len(r['frame_numbers'])):
                        frame_number = r['frame_numbers'][j]
                        
                        # Initialize the camera entry in all_frames if it doesn't exist
                        if i not in all_frames:
                            all_frames[i] = {}
                        
                        # Initialize the vehicle entry in the camera if it doesn't exist
                        if vehicle_id not in all_frames[i]:
                            all_frames[i][vehicle_id] = {}
                        
                        # Append the frame data to the vehicle entry for this camera
                        all_frames[i][vehicle_id].update(
                            {frame_number: {
                            'frame_number': frame_number,
                            'bounding_box': r['bboxes'][j],
                            'features': r['features'][j],
                            'compressed_image': None,
                            'shape': r['shapes'][j],
                            'timestamp': r['timestamps'][j],}
                            })
    elif reid_DB:
        # Gather all frames from the Database, separated by vehicle ID
        # For each vehicle ID, we will have a list of frames
        # We will then run the ReID model on each frame and compare the features
        # If the features are similar, we will consider them as the same vehicle
        # If the features are different, we will consider them as different vehicles
        all_frames = reid_DB.get_all_frames()
    else:
        logger.error("No valid input provided. Please provide either a list of Pickle paths or a valid Database connection.")
        raise Exception()

    logger.info(f"Frames gathered. Took {time.time() - log_start_time:.3f} s.")

    # =============== UNIFY DIFFERENT IDs BOUNDING BOXES FROM DIFFERENT CAMERAs =============== #

    # all_frames must be constructed in this way:
    # - List of Track objects, where in each Track object, we have the following:
    #   - ID of the vehicle
    #   - Mean Feature of the vehicle
    #   - Global Start Time of the vehicle trajectory
    #   - Global End Time of the vehicle trajectory
    all_frames_tracks = []
    
    # Iterate through each camera
    logger.info("Constructing Track objects...")
    for i, cam in enumerate(all_frames):
        for vehicle in all_frames[cam]:
            # Get the frames for the vehicle
            frames = all_frames[cam][vehicle]

            # Gather the key of the first and last frame
            first_frame_index = list(frames.keys())[0]
            last_frame_index = list(frames.keys())[-1]

            # Frame of the first and last timestamps
            try: # DB format
                first_frame_index = int(first_frame_index.split("F")[-1])
                last_frame_index = int(last_frame_index.split("F")[-1])
            except: # Pickle format
                first_frame_index = int(first_frame_index)
                last_frame_index = int(last_frame_index)

            # Insert Track object into the list
            tracklet = Track(vehicle_id=vehicle)
            tracklet.id = len(all_frames_tracks)
            tracklet.cam = i
            tracklet.add_frames(frames)
            tracklet.extract_bboxes()
            tracklet.extract_features()

            # Set the start and end times of the tracklet
            if cams:
                tracklet.set_start_time(first_frame_index / cams.fps[i] / cams.scales[i] + cams.offset[i])
                tracklet.set_end_time(last_frame_index / cams.fps[i] / cams.scales[i] + cams.offset[i])
            else:
                logger.error("Camera Layout not provided. Cannot set start and end times.")
                raise Exception()

            all_frames_tracks.append(tracklet)

    n = len(all_frames_tracks)

    # precompute compatibility between tracks
    compat = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i, n):
            is_comp = tracks_compatible(all_frames_tracks[i], all_frames_tracks[j], cams)
            compat[i, j] = is_comp
            compat[j, i] = is_comp

    SIMILARITY_THRESHOLD = cfg.REID.TEST.SIMILARITY_THRESHOLD
    linkage = cfg.REID.TEST.LINKAGE_METHOD # 'mean_feature' / 'average' / 'single' / 'complete'
    last_mod = [0] * n # timestamp of when the last time an mtrack was modified
    timestamp = 1

    # Compute the similarity matrix between the mean features
    log_start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f = torch.Tensor(np.stack([track.mean_feature(method=similarity_method) for track in all_frames_tracks])).to(device)

    # Compute similarity matrix
    sim_matrix = torch.matmul(f, f.T).cpu().numpy()

    logger.info(f"Similarity Matrix computed. Took {time.time() - log_start_time:.3f} s.")

    # Create a DataFrame to store the similarity matrix
    df = pd.DataFrame(sim_matrix,
                      index=[track.id for track in all_frames_tracks],
                      columns=[track.id for track in all_frames_tracks])

    # MulticamTracks
    mtracks = [Trajectory(i, [all_frames_tracks[i]], len(all_frames)) for i in range(n)]

    # Simply, now we do the following:
    #   - For each row and column by column:
    #    - Check if the two tracks can be compatible (in terms of camera)
    #    - If they are compatible, we check if the similarity is above the MAX_SIMILARITY_THRESHOLD
    #    - If it is, we consider them as the same vehicle
    import heapq

    # Priority queue for merge candidates
    merge_queue = []

    # Keep track of merged trajectories
    remaining_tracks = set(range(len(df)))

    logger.info("Starting the merging process...")
    log_start_time = time.time()
    for i in range(len(df)):
        for j in range(i + 1, len(df)):  # Avoid duplicates and self-pairs
            # First of all we control the cameras, if they are compatible
            if tracks_compatible(all_frames_tracks[i], all_frames_tracks[j], cams) and df.iloc[i, j] >= SIMILARITY_THRESHOLD:  # Check similarity threshold
                merge_queue.append((-df.iloc[i, j], timestamp, i, j))  # Use -similarity for max-heap

    heapq.heapify(merge_queue)

    while merge_queue:
        minus_sim, t, i, j = heapq.heappop(merge_queue)

        # Skip if trajectories are no longer valid (e.g., already merged)
        if i not in remaining_tracks or j not in remaining_tracks:
            continue
        
        # This means that we do break the loop if we arrived in a point where the similarity is below the given threshold
        if minus_sim > -SIMILARITY_THRESHOLD:
            break

        # if since the queue entry any tracks was modified, it is an invalid entry
        if t < max(last_mod[i], last_mod[j]):
            continue

        # Merge logic: Combine metadata, update mean features, etc.
        mtracks[i].merge_with(mtracks[j])
        timestamp += 1 # Update timestamp
        last_mod[i] = timestamp
        last_mod[j] = timestamp

        # Remove the merged trajectory
        remaining_tracks.remove(j)

        # Update similarity matrix for the merged trajectory
        for k in remaining_tracks:
            if k != i:
                track1 = mtracks[i]
                track2 = mtracks[k]

                # Check camera compatibility
                if have_mutual_cams(track1, track2) or not any_compatible(track1, track2, cams):
                    continue
                
                # Compute new similarity               
                new_similarity = compute_similarity(track1, track2, linkage, True, df)
                if new_similarity >= SIMILARITY_THRESHOLD:
                    heapq.heappush(merge_queue, (-new_similarity, timestamp, i, k))

    final_trajectories = [mtracks[i] for i in remaining_tracks]

    for idx, mtrack in enumerate(final_trajectories):
        mtrack.id = idx
        for track in mtrack.tracks:
            track.id = idx

    logger.info("Merging process completed. Took {:.3f} s.".format(time.time() - log_start_time))

    # Insert the single camera tracks into a list divided by camera
    camera_tracks = [[] for _ in range(cams.n_cams)]
    for mtrack in final_trajectories:
        for track in mtrack.tracks:
            camera_tracks[track.get_cam_id()].append(track)

    # Run MOTA metrics
    pickle_global_path = cfg_file.MTMC.PICKLE_GLOBAL_PATH
    for i, cam_track_list in tqdm(enumerate(camera_tracks), desc="Saving final MTMC predictions", unit="Camera", total=len(camera_tracks)):
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = f'MTMC-predictions_camera-{i}_{date}.txt'

        pickle_tracklets = []
        res = {
            "frame": [],
            "track_id": [],
            "bbox_topleft_x": [],
            "bbox_topleft_y": [],
            "bbox_width": [],
            "bbox_height": [],
        }

        # Retrieve the tracks for the camera
        for track in cam_track_list:
            # frames:
            #  - [0] => frame_number
            #  - [1] => bounding box
            #  - [2] => features
            #  - [3] => compressed_image
            #  - [4] => shape
            #  - [5] => timestamp
            frames = track.frames

            try:
                frame_numbers = [frame[0] for k, frame in frames.items()]
                bboxes = [frame[1] for k, frame in frames.items()]
                features = [frame[2] for k, frame in frames.items()]
            except:
                frame_numbers = [frame['frame_number'] for k, frame in frames.items()]
                bboxes = [frame['bounding_box'] for k, frame in frames.items()]
                features = [frame['features'] for k, frame in frames.items()]

            # Write formatted output to the file
            # ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Conf', 'Xworld', 'Yworld', 'Zworld']
            for bbox, frame_num in zip(bboxes, frame_numbers):
                tx, ty, w, h = bbox
                res['frame'].append(frame_num)
                res['track_id'].append(track.id)
                res['bbox_topleft_x'].append(int(tx))
                res['bbox_topleft_y'].append(int(ty))
                res['bbox_width'].append(int(round(w)))
                res['bbox_height'].append(int(round(h)))

            # Append to pickle
            pickle_tracklets.append({
                'id': id,
                'features': features,
                # 'shapes': shapes,
                'frame_numbers': frame_numbers,
                'bboxes': bboxes,
                # 'timestamps': timestamps,
                'vehicle_id': track.vehicle_id,
            })

        # MOTChallenge format for Dataframe
        df = pd.DataFrame(res)
        df.columns   = ['frame', 'track_id', 'bbox_topleft_x', 'bbox_topleft_y', 'bbox_width', 'bbox_height']
        df['conf']   = 1
        df['Xworld'] = -1
        df['Yworld'] = -1
        df['Zworld'] = -1
        df.to_csv(output_file, index=False, header=False)

        # Export the pickle file
        pickle_path = os.path.join(pickle_global_path, f"multitracklet_cam{i}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_tracklets, f)

    if cfg_file.MTMC.SHOW_MTMC_IMAGES == True:
        # We do need to show some results starting from the camera_tracks
        num_vehicles_to_show = 5
        min_bbox_area = 100
        col_names, row_names = [], []
        col_imgs, row_imgs = {}, {}

        # get a random seed each time (for the random sampling here)
        random.seed(datetime.now())

        # take a random sample of vehicles based on num_vehicles_to_show
        random_vehicles = random.sample(final_trajectories, num_vehicles_to_show)

        # Iterate through each camera
        for i, traj in enumerate(random_vehicles):
            # Retrieve the list of tracks for this trajectory
            cam_tracks = traj.tracks

            first_track = cam_tracks[0]

            found = False
            # For the first track, we take the index of the frame which has a shape of at least 80 in either width or height
            for k, v in first_track.frames.items():
                try:
                    shape = v[4]
                except:
                    shape = v['shape']
                if shape[0] >= min_bbox_area or shape[1] >= min_bbox_area:
                    idx = k
                    found = True
                    break

            if not found:
                idx = int(len(first_track.frames) / 2)
                idx = list(first_track.frames.keys())[idx]

            # Retrieve the middle frame index of the first track, the rest will be placed as columns
            q_frame = first_track.frames[idx]

            # q_frame[3] => compressed_image
            # q_frame[4] => shape
            try: # DB format
                comp_img = q_frame['compressed_image']
                shape = q_frame['shape']
            except: # Pickle format
                comp_img = q_frame[3]
                shape = q_frame[4]
            q_image = decompress_img(comp_img, shape, dtype=np.uint8)
            # reshape this numpy array to a PIL Image
            q_image = Image.fromarray(q_image).convert("RGB")
            q_image = ImageOps.contain(q_image, (128, 128))

            q_name = f"Q_{cam_tracks[0].vehicle_id}"
            row_names.append(q_name)

            if q_name not in row_imgs:
                row_imgs[q_name] = []
            row_imgs[q_name].append(q_image)

            # Simply loop through the rest of the tracks (avoiding the first one), take the middle frame and append it to the columns
            for j, track in enumerate(cam_tracks[1:]):

                found = False
                # For the first track, we take the index of the frame which has a shape of at least 80 in either width or height
                for k, v in track.frames.items():
                    try:
                        shape = v[4]
                    except:
                        shape = v['shape']
                    if shape[0] >= min_bbox_area or shape[1] >= min_bbox_area:
                        idx = k
                        found = True
                        break
                
                if not found:
                    idx = int(len(track.frames) / 2)
                    idx = list(track.frames.keys())[idx]

                g_frame = track.frames[idx]

                # q_frame[3] => compressed_image
                # q_frame[4] => shape
                try: # DB format
                    comp_img = g_frame['compressed_image']
                    shape = g_frame['shape']
                except: # Pickle format
                    comp_img = g_frame[3]
                    shape = g_frame[4]
                g_image = decompress_img(comp_img, shape, dtype=np.uint8)
                g_image = Image.fromarray(g_image).convert("RGB")
                g_image = ImageOps.contain(g_image, (128, 128))

                g_name = f"G_{track.vehicle_id}"
                col_names.append(g_name)

                if q_name not in col_imgs:
                    col_imgs[q_name] = []
                col_imgs[q_name].append(g_image)

        # take the maximum number of gallery images from the col_imgs dictionary
        K = max([len(imgs) for imgs in col_imgs.values()])

        fig, axes = plt.subplots(num_vehicles_to_show, K+1, figsize=(12, 15))
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

        # Iterate over the col_imgs and row_imgs dictionaries
        for i, (name, imgs) in enumerate(row_imgs.items()):
            query_name = name
            query_img = query_trans(imgs[0])

            # Safely remove all axis from the plot
            for ax in axes[i]:
                ax.axis("off")

            # Display the query image
            ax = axes[i, 0]
            ax.imshow(query_img)
                
            # Get the tracks related to the query image
            related_tracks = col_imgs.get(query_name, None)
            if related_tracks is not None:
                for j, img in enumerate(related_tracks):
                    ax = axes[i, j+1]
                    img = good_trans(img)
                    ax.imshow(img)
            else:
                for j in range(0, K):
                    ax = axes[i, j+1]
                    ax.imshow(transparent_img)

        plt.tight_layout()
        plt.show()
        plt.pause(0)

    logger.info("MTMC successfully completed.")

    # =============== TARGET VEHICLE SEARCH =============== #
    if cfg_file.MISC.RUN_TARGET_SEARCH == True:
        logger.info("Starting Target Vehicle search...")
        from target.target import target_search

        # Target image and camera ID
        target_path, target_cam = cfg_file.MISC.TARGET_PATH
        target = {"path": target_path, "camera": target_cam}
        target_search(target, final_trajectories, all_frames_tracks, cams, cfg)
    # ======================================================