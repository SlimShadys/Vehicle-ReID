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
# - Update trajectories, bounding boxes, and vehicles in the database           -> 50% (only for same camera for now)
# - Create Graphs for the trajectories                                          -> 0%
# - Create a GUI for the pipeline                                               -> 0%

import argparse
import os

import cv2
import imageio.v3 as iio
import networkx as nx
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from config import _C as cfg_file
from db.database import Database
from graph.model import GraphEngine
from misc.printer import Logger
from misc.utils import (compress_img, create_mask, decompress_img,
                        euclidean_dist, load_middle_frame, set_seed)
from reid.datasets.transforms import Transformations
from reid.model import ModelBuilder
from tracking.filter import filter_frames
from tracking.model import load_yolo

# Number of classes in each dataset (Only for ID classification tasks, hence only training set is considered)
num_classes = {
    'veri_776': 576,
    'veri_wild': 30671,
    'vehicle_id': 13164,
    'vru': 7086,
}

def sample_trajectory_frames(frames, num_samples=50):
    if len(frames) <= num_samples:
        return frames
    
    # Take the first frame, some middle frames, and the last frame
    step = len(frames) // (num_samples - 2)  # Number of frames to skip between samples
    sampled_frames = [frames[0]] + frames[step:len(frames)-step:step] + [frames[-1]]
    
    return sampled_frames

def main():
    parser = argparse.ArgumentParser(description="Vehicle Re-ID Pipeline")
    parser.add_argument("--config_file", default="", help="Path to config file", type=str)
    args = parser.parse_args()

    # Load the config file
    if args.config_file != "":
        cfg_file.merge_from_file(args.config_file)
    #cfg_file.freeze()
    
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
    set_seed(seed)

    # ========================== LOAD YOLO MODEL ========================== #
    yolo_model = load_yolo(tracking_cfg)
    input_path = tracking_cfg.VIDEO_PATH
    tracked_classes = tracking_cfg.MODEL.TRACKED_CLASS
    conf_thresh = tracking_cfg.MODEL.CONFIDENCE_THRESHOLD
    yolo_img_size = tracking_cfg.MODEL.YOLO_IMAGE_SIZE
    yolo_iou_threshold = tracking_cfg.MODEL.YOLO_IOU_THRESHOLD
    use_agnostic_nms = tracking_cfg.MODEL.USE_AGNOSTIC_NMS
    tracker_yaml_file = tracking_cfg.MODEL.YOLO_TRACKER
    use_roi_mask = tracking_cfg.USE_ROI_MASK
    roi_mask_path = tracking_cfg.ROI_MASK_PATH

    # ========================== LOAD REID MODEL ========================== #
    model_name = reid_cfg.MODEL.NAME
    pretrained = reid_cfg.MODEL.PRETRAINED
    dataset_name = reid_cfg.DATASET.DATASET_NAME
    model_configs = reid_cfg.MODEL
    color_model_configs = reid_cfg.COLOR_MODEL
    model_val_path = reid_cfg.TEST.MODEL_VAL_PATH

    # Load Logger
    logger = Logger()

    print("--------------------")
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

    # Transformations needed for the ReID model
    transforms = Transformations(dataset=None, configs=reid_cfg.AUGMENTATION)

    # ========================== LOAD COLOR MODEL ========================== #
    logger.color(f"Loading color model from {color_model_configs.NAME}...")
    model_builder = ModelBuilder(
        model_name=color_model_configs.NAME,
        pretrained=pretrained,
        num_classes=num_classes[dataset_name],
        model_configs=color_model_configs,
        device=device
    )
    car_classifier = model_builder.move_to(device)
    logger.color(f"Successfully loaded color model from {color_model_configs.NAME}!")
    # ======================================================================
        
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
    reader = iio.imopen(input_path, 'r', plugin='pyav')

    # Get some Tracking Configs
    STATIONARY_THRESHOLD = tracking_cfg.STATIONARY_THRESHOLD    # Define a threshold for IoU below which we consider the vehicle as moving
    MIN_STATIONARY_FRAMES = tracking_cfg.MIN_STATIONARY_FRAMES  # Minimum number of frames a vehicle must be stationary to be considered stationary
    MIN_BOX_SIZE = tracking_cfg.MIN_BOX_SIZE                    # Minimum width or height of a bounding box to be considered complete
    TARGET_COLORS = misc_cfg.TARGET_COLOR                       # Define the target colors for the color model

    # Initialize dictionary to hold the previous frame's bounding boxes, classes, confidences, and cropped images
    chunk_dict = {}  # Temporary dictionary for chunk storage
    chunk_size = 2000  # Define the number of frames per chunk

    # Initialize various counters counter
    total_frames_removed = 0 # Counter for total number of frames removed
    color_removed = 0
    stationary_removed = 0
    incomplete_removed = 0
    total_frames = 0

    # DB connection and insertion
    clean_db = db_cfg.CLEAN_DB
    reid_DB = Database(db_configs=db_cfg, logger=logger)

    if clean_db:
        logger.db("Cleaning the database...")
        reid_DB.clean()

    # logger.db("Inserting Cameras into the database...")
    camera0 = {'_id': 0, 'name': 'Camera-0',
               'location': 'Rome', 'coordinates': [41.9028, 12.4964],
               'description': 'Traffic - Left side - Tecnopolo'}
    #reid_DB.insert_camera(camera0)

    camera1 = {'_id': 1, 'name': 'Camera-1',
                'location': 'Rome', 'coordinates': [41.9028, 12.4964],
                'description': 'Traffic - Right side - Tecnopolo'}
    #reid_DB.insert_camera(camera1)

    # Instantiate the Global camera
    camera = camera0

    if (use_roi_mask == True):
        # Load the mask
        if (roi_mask_path != ""):
            mask = cv2.imread(roi_mask_path, 0)  # Grayscale mask
        else:
            # We need to extract a single frame from the video to create the mask
            mask_frame = iio.imread(input_path, index=0, plugin="pyav")
            mask = create_mask(mask_frame, input_path)

        mask = cv2.resize(mask, (FRAME_WIDTH, FRAME_HEIGHT))
        logger.info(f"Successfully loaded ROI mask")

    if misc_cfg.RUN_PIPELINE:
        # Initialize the progress bar with the total number of frames
        with tqdm(total=total_n_frames, desc="Processing Video", unit="frame") as pbar:
            # Run the tracking loop
            for frame_number, orig_frame in enumerate(reader.iter(), start=1):

                # Calculate the timestamp for the current frame
                timestamp = frame_number / video_FPS

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

                # Extract bounding boxes, classes, names, and confidences
                boxes = results[0].boxes.xyxy.tolist()
                classes = results[0].boxes.cls.tolist()
                ids = results[0].boxes.id.tolist() if results[0].boxes.id is not None else [0] * len(classes)
                names = results[0].names
                confidences = results[0].boxes.conf.tolist()

                # Iterate over detected objects
                for box, cls, id, conf in zip(boxes, classes, ids, confidences):
                    x1, y1, x2, y2 = map(int, box)  # Convert to integers
                    # confidence = conf
                    detected_class = int(cls)
                    name = names[detected_class]
                    id = int(id)
                    crop = orig_frame[y1:y2, x1:x2]  # Crop the bounding box from the image

                    # Add frame info to the chunk dictionary
                    if id not in chunk_dict:
                        chunk_dict[id] = []

                    chunk_dict[id].append({
                        'class': name,
                        'confidence': conf,
                        'bounding_box': box,
                        'cropped_image': crop,
                        'timestamp': round(timestamp, 2)  # Round to 2 decimal places
                    })

                # Flush and reset the dictionary after processing a chunk
                if frame_number % chunk_size == 0 or frame_number == total_n_frames:
                    total_frames += sum(len(v) for v in chunk_dict.values())  # Increment total frames counter

                    # Filter frames in chunk_dict before storing
                    filtered_chunk_dict, frames_removed = filter_frames(chunk_dict, model, car_classifier, transforms, device,
                                                                        TARGET_COLORS, STATIONARY_THRESHOLD, MIN_STATIONARY_FRAMES,
                                                                        MIN_BOX_SIZE, FRAME_WIDTH, FRAME_HEIGHT)
                    total_frames_removed += sum(frames_removed.values())
                    color_removed += frames_removed['color']
                    stationary_removed += frames_removed['stationary']
                    incomplete_removed += frames_removed['incomplete']

                    # Before running the ReID model and returning the features, we must update the class of the bounding boxes
                    # We simply take the number of occurences of each class and assign the most frequent one to the bounding box
                    for id in filtered_chunk_dict:
                        frames = filtered_chunk_dict[id]
                        classes = [frame['class'] for frame in frames]
                        most_common_class = max(set(classes), key=classes.count)

                        # Remove all the classes occurrences from the frames, since we have a global variable for the class
                        for frame in frames: del frame['class']

                        filtered_chunk_dict[id] = {'class': most_common_class,
                                                'frames': [frame for frame in frames],
                                                #'possible_different_vehicles': filtered_different_chunk_dict.get(id, [])
                                                }

                    for id in filtered_chunk_dict:
                        # Sample frames from the trajectory to reduce the number of frames stored
                        filtered_chunk_dict[id]['frames'] = sample_trajectory_frames(filtered_chunk_dict[id]['frames'], num_samples=50)

                    # Iterate through each ID in filtered_full_dict
                    if tracking_cfg.SAVE_BOUNDING_BOXES:
                        for id in filtered_chunk_dict:
                            # Create a directory for the ID if it doesn't exist
                            folder_path = os.path.join(os.path.join(tracking_cfg.BOUNDING_BOXES_OUTPUT_PATH, camera['name'], f"{id}"))
                            if not os.path.exists(folder_path):
                                os.makedirs(folder_path)

                            # Extract the cropped images for each frame
                            frames = [frame['cropped_image'] for frame in filtered_chunk_dict[id]['frames']]

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

                    # Run ReID model on the remaining frames
                    for id in tqdm(filtered_chunk_dict, desc="Running ReID model on remaining vehicles"):
                        frames = [frame['cropped_image'] for frame in filtered_chunk_dict[id]['frames']]

                        # Run the ReID model on the frames
                        for i, orig_frame in enumerate(frames):
                            frame = Image.fromarray(orig_frame).convert('RGB')  # Convert to PIL Image
                            frame = transforms.val_transform(frame)             # Convert to torch.Tensor
                            frame = frame.unsqueeze(0).float().to(device)

                            features = model.forward(frame, training=False)

                            # Add this feature to the dictionary for the ID
                            # id -> {class, frames: [{frame1}, {frame2}, ...]}
                            filtered_chunk_dict[id]['frames'][i]['features'] = features.detach().cpu().numpy()

                    # Insert the data into the MongoDB database
                    logger.db("Inserting Vehicles, Trajectories & BBoxes into the database...")
                    for id in filtered_chunk_dict:
                        # Combine Camera ID with Vehicle ID to create a unique ID for the vehicle
                        unique_vehicle_id = f"CAM{camera['_id']}_V{id}"

                        # Combine Vehicle ID with Trajectory ID
                        # Since we have one trajectory per vehicle, we can use the ID of the vehicle as the ID of the trajectory
                        # along with the suffix '_0' to indicate the first trajectory of the vehicle:
                        #  -> CAM0_V10_T0 : Trajectory 0 of Vehicle 10 from Camera 0
                        #  -> CAM0_V22_T0 : Trajectory 0 of Vehicle 22 from Camera 0
                        #  -> CAM1_V173_T0: Trajectory 0 of Vehicle 173 from Camera 1
                        trajectory_id = unique_vehicle_id + "_T0"

                        # It means that the trajectory already exists in the database
                        # Hence, we should update the trajectory with new frames instead of inserting a new one
                        if reid_DB.trajectories_col.find_one({'_id': trajectory_id}):
                            update = True
                        else:
                            update = False    

                        # Check if the vehicle already exists. This could happen if sometimes, the vehicle is
                        # being detected between two different chunks of frames.
                        # For example: In the first chunk, the vehicle is detected at frame 100, and in the second chunk,
                        # the vehicle is detected at frame 2151. In this case, that vehicle was being added to the database
                        # already at frame 100.
                        vehicle_class = filtered_chunk_dict[id]['class'] # Get the class of the vehicle
                        if not reid_DB.vehicles_col.find_one({'_id': unique_vehicle_id}):
                            reid_DB.insert_vehicle({'_id': unique_vehicle_id, 'class': vehicle_class})
                        else:
                            logger.db(f"Vehicle with ID {unique_vehicle_id} already exists, skipping insertion.")
                    
                        filtered_chunk_dict[id]['_id'] = trajectory_id              # ID of the trajectory
                        filtered_chunk_dict[id]['vehicle_id'] = unique_vehicle_id   # ID of the vehicle
                        filtered_chunk_dict[id]['camera_id'] = camera['_id']        # ID of the camera

                        # Compress both the cropped imgs and the features into a suitable format for MongoDB
                        frames = filtered_chunk_dict[id]['frames']
                        for frame in frames:
                            # Compress the cropped image
                            compressed_data = compress_img(frame['cropped_image'])
                            shape = frame['cropped_image'].shape
                            features = frame['features'].tolist()[0]
                            
                            # From now on, everything has been compressed, so insert it back to the Dict and start storing it
                            frame['cropped_image'] = compressed_data
                            frame['shape'] = shape
                            frame['features'] = features

                        for j, frame in enumerate(frames):
                            # Create a unique ID for the bounding box
                            # Format: CAM0_V10_F0/1/2/3/... (Camera 0, Vehicle 10, Frame 0/1/2/3/...)
                            frame['_id'] = unique_vehicle_id + f"_F{j}"
                            frame['vehicle_id'] = unique_vehicle_id

                            if not reid_DB.bboxes_col.find_one({'_id': frame['_id']}):
                                reid_DB.insert_bbox(frame)
                            else:
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
                                logger.db(f"Vehicle with ID {unique_vehicle_id} already exists, skipping insertion.")

                        # We need the start time and the ending time, which we can get from the timestamps
                        # of the first and last frames
                        start_time = frames[0]['timestamp']
                        end_time = frames[-1]['timestamp']

                        filtered_chunk_dict[id]['start_time'] = start_time
                        filtered_chunk_dict[id]['end_time'] = end_time

                        # Check if there are possible different vehicles to store
                        #possible_different_vehicles = filtered_chunk_dict[id].get('possible_different_vehicles', [])

                        reid_DB.insert_trajectory({
                            '_id': filtered_chunk_dict[id]['_id'],
                            'vehicle_id': filtered_chunk_dict[id]['vehicle_id'],
                            'camera_id': filtered_chunk_dict[id]['camera_id'],
                            'start_time': start_time,
                            'end_time': end_time,
                            'trajectory_data': frames,
                            #'possible_different_vehicles': possible_different_vehicles
                        }, update=update)

                    # After the chunk is stored, Python's garbage collector will handle memory,
                    # but we can also explicitly free memory if necessary.
                    del chunk_dict
                    torch.cuda.empty_cache()  # If running on GPU, clear cache to free up memory
                    chunk_dict = {}  # Clear the dictionary to free memory

                # Update the progress bar after processing each frame
                pbar.update(1)

        # Release video capture and destroy all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

        print("===============================================")
        logger.info("Pipeline completed successfully! Here's a summary of the filtering")
        logger.info(f"\t- Total BBoxes: {total_frames}")
        logger.info(f"\t- BBoxes removed: {total_frames_removed}")
        logger.info(f"\t\t- Frames removed due to color: {color_removed}")
        logger.info(f"\t\t- Frames removed due to stationary vehicles: {stationary_removed}")
        logger.info(f"\t\t- Frames removed due to incomplete bounding boxes: {incomplete_removed}")
        logger.info(f"\t- Percentage removed: {(total_frames_removed / total_frames) * 100:.2f}%")
        print("===============================================")
    
    elif misc_cfg.RUN_UNIFICATION:
        # Get the similarity method to use
        similarity_algorithm    = reid_cfg.TEST.SIMILARITY_ALGORITHM    # 'cosine' / 'euclidean'
        similarity_method       = reid_cfg.TEST.SIMILARITY_METHOD       # 'individual' / 'mean'

        # =============== UNIFY DIFFERENT IDs BOUNDING BOXES FROM SAME CAMERA =============== # Taking the MEAN of the embeddings
        # # After the pipeline is done, we must refine the BBoxes to:
        # # - Unify the BBoxes of the same vehicle that have been categorized as different vehicles
        # #   -> We can think of using the ReID model to check if the BBoxes are from the same vehicle
        camera_id = camera['_id']
        camera_vehicles = reid_DB.get_camera_frames(camera_id=camera_id)

        # Initialize dictionary to store the max similarity for each vehicle
        max_similarity_per_vehicle = {}

        if similarity_method == 'mean':
            # Construct a similarity matrix for the vehicles from the same camera
            # We will compare the features of the vehicles from the same camera
            # If the features are similar, we will consider them as the same vehicle
            # If the features are different, we will consider them as different vehicles
            mean_embeddings_dict = {}

            # Extract embeddings and compute mean per vehicle
            for vehicle, frames in camera_vehicles.items():
                # Save mean embedding for later comparison
                mean_embeddings_dict[(camera_id, vehicle)] = torch.stack([torch.tensor(frame) for frame in [frames[id][0] for id in frames]]) \
                                                            .to(device) \
                                                            .mean(dim=0)  # Shape [2048]

            # Convert embeddings to tensor and normalize if needed
            vehicle_ids = list(mean_embeddings_dict.keys())
            mean_embeddings = torch.stack([mean_embeddings_dict[vid] for vid in vehicle_ids])

            # Normalize the embeddings if necessary
            if reid_cfg.TEST.NORMALIZE_EMBEDDINGS:
                mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)

            # Number of vehicles
            num_vehicles = len(vehicle_ids)

            # Compute pairwise similarity for vehicles from different cameras
            for i in range(num_vehicles):
                camera_i, vehicle_i = vehicle_ids[i]
                max_similarity = -float('inf')  # Initialize with very low similarity
                max_vehicle_pair = None  # Store the pair with the highest similarity

                for j in range(i+1, num_vehicles):  # Only compute upper triangle
                    camera_j, vehicle_j = vehicle_ids[j]

                    if similarity_algorithm == 'euclidean':
                        similarity = euclidean_dist(mean_embeddings[i].unsqueeze(0), mean_embeddings[j].unsqueeze(0), train=False)
                    elif similarity_algorithm == 'cosine':
                        similarity = torch.nn.functional.cosine_similarity(mean_embeddings[i], mean_embeddings[j], dim=0).item()

                    # Check if this is the highest similarity encountered for this vehicle
                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_vehicle_pair = vehicle_j

                # Store the vehicle pair with the highest similarity
                if max_vehicle_pair:
                    max_similarity_per_vehicle[vehicle_i] = (max_vehicle_pair, max_similarity)

        # =============== UNIFY DIFFERENT IDs BOUNDING BOXES FROM SAME CAMERA ================= #
        elif similarity_method == 'individual':
            # Vehicle IDs list
            vehicle_ids = list(camera_vehicles.keys())

            # Number of vehicles
            num_vehicles = len(vehicle_ids)

            # Normalize the embeddings frmo camera_vehicles
            for vehicle in camera_vehicles:
                for frame in camera_vehicles[vehicle]:
                    camera_vehicles[vehicle][frame][0] = torch.nn.functional.normalize(torch.tensor(camera_vehicles[vehicle][frame][0]), p=2, dim=0).tolist()

            # Compare all vehicle pairs
            for i in tqdm(range(num_vehicles), desc="Computing Similarity Matrix"):
                vehicle_i = vehicle_ids[i]
                max_similarity = -float('inf')  # Initialize with very low similarity
                max_vehicle_pair = None         # Store the pair with the highest similarity

                for j in range(i + 1, num_vehicles):  # Only compute upper triangle
                    vehicle_j = vehicle_ids[j]

                    similarities = []  # Store individual frame-to-frame similarities

                    # Compare each frame of vehicle_i with each frame of vehicle_j
                    for frame_i in camera_vehicles[vehicle_i]:
                        for frame_j in camera_vehicles[vehicle_j]:
                            emb_i = torch.tensor(camera_vehicles[vehicle_i][frame_i][0]).to(device)  # [2048]
                            emb_j = torch.tensor(camera_vehicles[vehicle_j][frame_j][0]).to(device)  # [2048]

                            # Compute the similarity based on the chosen method
                            if similarity_algorithm == 'euclidean':
                                similarity = euclidean_dist(emb_i.unsqueeze(0), emb_j.unsqueeze(0), train=False).item()
                            elif similarity_algorithm == 'cosine':
                                similarity = torch.nn.functional.cosine_similarity(emb_i, emb_j, dim=0).item()

                            similarities.append(similarity)

                    # Compute the average similarity between the two vehicles
                    avg_similarity = np.mean(similarities)

                    # Check if this is the highest similarity encountered for this vehicle
                    if avg_similarity > max_similarity:
                        max_similarity = avg_similarity
                        max_vehicle_pair = vehicle_j

                # Store the vehicle pair with the highest similarity
                if max_vehicle_pair:
                    max_similarity_per_vehicle[(vehicle_i)] = (max_vehicle_pair, max_similarity)
        else:
            raise ValueError(f"Invalid similarity method: {similarity_method}")

        # Create a graph where nodes are vehicle IDs
        G = nx.Graph()

        # Print out vehicle pairs with the highest similarity per vehicle
        if max_similarity_per_vehicle:
            print("Vehicle pairs with the highest similarity (across cameras):")
            for veh1, (veh2, similarity) in max_similarity_per_vehicle.items():
                G.add_node(veh1)  # Add the vehicle (bbox_id) as a node
                if similarity >= 0.80:
                    print(f"\tVehicle {veh1} is most similar to Vehicle {veh2} with similarity {similarity:.2f}")
                    # In this case, we must update the DB to unify the IDs, the trajectories, and the bounding boxes
                    # We must also update the bounding boxes to reflect the new vehicle ID
                    # We must also update the trajectories to reflect the new vehicle ID
                    G.add_edge(veh1, veh2)  # Add an edge if similarity exceeds threshold
        else:
            print("No similar vehicles detected across cameras.")

        # Find all connected components in the graph
        connected_components = list(nx.connected_components(G))

        # Merge vehicles based on connected components
        for component in connected_components:
            component_list = list(component)
            if len(component_list) > 1:
                component_list = sorted(component_list)
                print(f"Merging vehicles: {component_list}")
                
                # Update the database by:
                # 1. Updating the bounding boxes
                # 2. Updating the trajectories
                # 3. Removing the vehicles from the database

                # primary_vehicle = component_list[0]  # Choose the first vehicle as the primary one
                # for other_vehicle in component_list[1:]:
                #     reid_DB.update_bbox(primary_vehicle, other_vehicle)
                #     reid_DB.update_trajectory(primary_vehicle, other_vehicle)
                #     reid_DB.remove_vehicle(other_vehicle)
        print(f"Single camera unification complete for Camera with ID: {camera['_id']}.")
        # =================================================================================== #

        return
    
        # =============== UNIFY DIFFERENT IDs BOUNDING BOXES FROM DIFFERENT CAMERAs =============== #
        # Gather all frames from the Database, separated by vehicle ID
        # For each vehicle ID, we will have a list of frames
        # We will then run the ReID model on each frame and compare the features
        # If the features are similar, we will consider them as the same vehicle
        # If the features are different, we will consider them as different vehicles
        all_frames = reid_DB.get_all_frames()

        mean_embeddings_dict = {}

        # Extract embeddings and compute mean per vehicle
        for camera_id, vehicles in all_frames.items():
            for vehicle_id, frames in vehicles.items():
                # Save mean embedding for later comparison
                mean_embeddings_dict[(camera_id, vehicle_id)] = torch.stack([torch.tensor(frames[x][0]) for x in frames]).to(device).mean(dim=0)  # Shape [2048]   

        # Convert embeddings to tensor and normalize if needed
        vehicle_ids = list(mean_embeddings_dict.keys())
        mean_embeddings = torch.stack([mean_embeddings_dict[vid] for vid in vehicle_ids])

        # Normalize the embeddings if necessary
        if reid_cfg.TEST.NORMALIZE_EMBEDDINGS:
            mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)

        # Number of vehicles divided per camera
        camera0_vehicle_ids = [id for id in vehicle_ids if id[0] == 0]
        camera1_vehicle_ids = [id for id in vehicle_ids if id[0] == 1]
        num_vehicles_camera0 = len(camera0_vehicle_ids)
        num_vehicles_camera1 = len(camera1_vehicle_ids)

        # Create a similarity matrix (vehicles x vehicles) only for vehicles from different cameras
        similarity_matrix = torch.zeros((num_vehicles_camera0, num_vehicles_camera1))

        # Store the maximum similarity per vehicle
        max_similarity_per_vehicle = {}

        # Compute pairwise similarity for vehicles from different cameras
        for i in range(num_vehicles_camera0):
            camera_i, vehicle_i = vehicle_ids[i]
            max_similarity = -float('inf')  # Initialize with very low similarity
            max_vehicle_pair = None  # Store the pair with the highest similarity

            for j in range(num_vehicles_camera1):
                mapping_index = j + num_vehicles_camera0
                camera_j, vehicle_j = vehicle_ids[mapping_index]

                # Only compare vehicles from different cameras
                if camera_i != camera_j:
                    if similarity_method == 'euclidean':
                        similarity = euclidean_dist(mean_embeddings[i].unsqueeze(0), mean_embeddings[mapping_index].unsqueeze(0), train=False)
                    elif similarity_method == 'cosine':
                        similarity = torch.nn.functional.cosine_similarity(mean_embeddings[i], mean_embeddings[mapping_index], dim=0).item()

                    # Fill both upper and lower triangle of the matrix
                    similarity_matrix[i, j] = similarity

                    # Check if this is the highest similarity encountered for this vehicle
                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_vehicle_pair = (camera_i, vehicle_i, camera_j, vehicle_j)

            # Store the vehicle pair with the highest similarity
            if max_vehicle_pair:
                max_similarity_per_vehicle[(camera_i, vehicle_i)] = (max_vehicle_pair, max_similarity)

        # Convert tensor to numpy array if it's not already
        if torch.is_tensor(similarity_matrix):
            similarity_matrix = similarity_matrix.cpu().numpy()

        # Plot the heatmap with the combined mask
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, 
                    cmap="YlGnBu",
                    vmin=-1, vmax=1,
                    annot=True,
                    fmt=".2f",
                    square=True,
                    xticklabels=[f"{cam}_{veh}" for cam, veh in camera1_vehicle_ids],
                    yticklabels=[f"{cam}_{veh}" for cam, veh in camera0_vehicle_ids])
        plt.title("Vehicle Embedding Similarity Matrix (Across Cameras)")
        plt.tight_layout()
        plt.show()

        # Print out vehicle pairs with the highest similarity per vehicle
        if max_similarity_per_vehicle:
            print("Vehicle pairs with the highest similarity (across cameras):")
            for (cam1, veh1), ((cam1, veh1, cam2, veh2), similarity) in max_similarity_per_vehicle.items():
                print(f"\tVehicle {veh1} from Camera {cam1} is most similar to Vehicle {veh2} from Camera {cam2} with similarity {similarity:.2f}")
        else:
            print("No similar vehicles detected across cameras.")

        # Extract the top k prediction from the similarity matrix computed earlier
        top_k_predictions = {vehicle_id: sorted([(camera1_vehicle_ids[j], similarity_matrix[i, j]) for j in range(num_vehicles_camera1)], key=lambda x: x[1], reverse=True) for i, vehicle_id in enumerate(camera0_vehicle_ids)}
        
        # Also add camera 1 vehicles to the predictions
        top_k_predictions.update({vehicle_id: sorted([(camera0_vehicle_ids[j], similarity_matrix[j, i]) for j in range(num_vehicles_camera0)], key=lambda x: x[1], reverse=True) for i, vehicle_id in enumerate(camera1_vehicle_ids)})

        K = 3
        vehicles_to_show = 10 # len(camera0_vehicle_ids)
        # camera0_vehicle_ids = camera0_vehicle_ids[vehicles_to_show:-1]
        fig, axes = plt.subplots(vehicles_to_show, K+1, figsize=(12, 15))

        query_trans = T.Pad(4, 0)
        good_trans = T.Pad(4, (0, 255, 0))
        bad_trans = T.Pad(4, (255, 0, 0))

        for i, (camera_id, vehicle_id) in enumerate(camera0_vehicle_ids):
            if i == vehicles_to_show: break

            query_img = load_middle_frame(tracking_cfg.BOUNDING_BOXES_OUTPUT_PATH, vehicle_id, camera_id)
            query_img = query_trans(query_img)
            ax = axes[i, 0]  # Display in the first column of the row
            ax.imshow(query_img)
            ax.axis("off")

            # Retrieve the top K predictions for the vehicle
            predictions = top_k_predictions[(camera_id, vehicle_id)]
            
            # Plot the top K predictions
            for j in range(min(K, len(predictions))):
                (pred_camera, pred_vehicle), similarity = predictions[j]
                img_pred = load_middle_frame(tracking_cfg.BOUNDING_BOXES_OUTPUT_PATH, pred_vehicle, pred_camera)
                if j == 0:
                    img_pred = good_trans(img_pred)
                else:
                    img_pred = bad_trans(img_pred)

                ax = axes[i, j+1]  # Display in subsequent columns
                ax.title.set_text(f"{similarity:.2f}")
                ax.imshow(img_pred)
                ax.axis("off")

        plt.tight_layout()
        plt.show()

        # # ========================================================================================= #
    else:
        print("No pipeline to run. Please check the configuration file.")
        exit(0)

        # # ========================== FEATURE MATCHING ========================== #
        # all_frames = reid_DB.get_all_frames()

        # # Feature matching variables
        # matching_method = 'bf'  # 'bf' or 'flann'
        # feature_method = 'orb'  # 'orb' or 'sift'

        # # ORB Settings
        # nfeatures = 10000
        # fast_threshold = 20
        # match_threshold = 1000 # Threshold for matches
        # distance_threshold = 50  # Distance threshold for determining good matches

        # # BF settings
        # norm_type = cv2.NORM_HAMMING  # For ORB descriptor
        # cross_check = True            # Cross-check for BFMatcher

        # # FLANN settings
        # index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        # search_params = dict(checks=200)  # Higher checks = more precision, slower

        # # Feature detector
        # if feature_method == 'orb':
        #     feature_descriptor = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=fast_threshold)
        # elif feature_method == 'sift':
        #     feature_descriptor = cv2.SIFT_create()
        # else:
        #     raise ValueError(f"Invalid method: {feature_method}")

        # # Matcher
        # if matching_method == 'flann':
        #     flann = cv2.FlannBasedMatcher(index_params, search_params)
        # elif matching_method == 'bf':
        #     bf = cv2.BFMatcher(normType=norm_type, crossCheck=cross_check)
        # else:
        #     raise ValueError(f"Invalid method: {matching_method}")

        # def extract_keypoints_and_descriptors(frame):
        #     """Extracts keypoints and descriptors using ORB."""
        #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     keypoints, descriptors = feature_descriptor.detectAndCompute(gray_frame, None)

        #     return keypoints, descriptors

        # def match_descriptors(desc1, desc2):
        #     """Matches two sets of descriptors using BFMatcher."""
        #     if desc1 is None or desc2 is None:
        #         return 0  # No matches if one of the descriptors is None

        #     if matching_method == 'bf':
        #         matches = bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))
        #         # Sort matches by distance
        #         matches = sorted(matches, key=lambda x: x.distance)
        #     elif matching_method == 'flann':
        #         matches = flann.knnMatch(desc1, desc2, k=3)  # KNN instead of brute force
        #         # Sort matches based on the first match's distance, but check if the tuple has at least one element
        #         matches = sorted(matches, key=lambda x: x[0].distance if len(x) > 0 else float('inf'))
        #     else:
        #         raise ValueError(f"Invalid method: {matching_method}")

        #     return matches

        # # Extract descriptors for each vehicle's frames
        # vehicle_matches = {}
        # vehicle_descriptors = {}

        # for camera, vehicle_id in tqdm(all_frames.items(), desc="Extracting features for vehicles"):
        #     # Extract descriptors for each vehicle
        #     for vehicle_id, frames in vehicle_id.items():
        #         descriptors_list = []
                
        #         for frame_data in frames:
        #             encoded_frame = frames[frame_data][1]
        #             shape = frames[frame_data][2]
        #             decoded_frame = decompress_img(encoded_frame, shape, np.uint8)
                    
        #             _, descriptors = extract_keypoints_and_descriptors(decoded_frame)
        #             if descriptors is not None:
        #                 descriptors_list.append(descriptors)

        #         # Store all descriptors of the vehicle
        #         if descriptors_list:
        #             vehicle_descriptors[vehicle_id] = np.vstack(descriptors_list)  # Stack all descriptors

        # # Compare vehicles with each other
        # vehicle_ids = list(vehicle_descriptors.keys())
        # num_vehicles = len(vehicle_ids)

        # # Loop through vehicle pairs
        # for i in tqdm(range(num_vehicles), desc="Comparing vehicles"):
        #     for j in range(i+1, num_vehicles):
                
        #         # Extract descriptors for the two vehicles
        #         id1, id2 = vehicle_ids[i], vehicle_ids[j]
        #         desc1, desc2 = vehicle_descriptors[id1], vehicle_descriptors[id2]

        #         # Match descriptors
        #         matches = match_descriptors(desc1, desc2)

        #         if matching_method == 'flann':
        #             # Filter out empty match tuples and only select good matches
        #             good_matches = [m[0] for m in matches if len(m) > 0 and m[0].distance < distance_threshold]
        #         elif matching_method == 'bf':
        #             # For BFMatcher case (without FLANN), no need to index the tuple
        #             # since it's already a match object
        #             good_matches = [m for m in matches if m.distance < distance_threshold]
        #         else:
        #             good_matches = [m for m in matches if m.distance < 0.7 * distance_threshold]
                
        #         if len(good_matches) >= match_threshold:
        #             vehicle_matches[(id1, id2)] = len(good_matches)
        #             #print(f"Vehicles {id1} and {id2} have {len(good_matches)} good matches and might be the same.")

        # # Display the matching results
        # if vehicle_matches:
        #     print("\nPotentially matching vehicles based on feature matching:")
        #     for (v1, v2), match_count in vehicle_matches.items():
        #         print(f"<{v1}, {v2}>: {match_count} good matches.")
        # else:
        #     print("No similar vehicles detected.")

# Usage: python main.py <path_to_config.yml>
if __name__ == '__main__':
    
    print("|***************************************************|")
    print("|                   Vehicle Re-ID                   |")
    print("|                      Pipeline                     |")
    print("|---------------------------------------------------|")
    print("|             Made by: Gianmarco Scarano            |")
    print("|       --- MSc Student at AI & Robotics ---        |")
    print("|      --- University of Rome, La Sapienza ---      |")
    print("|         ---        Rome, Italy        ---         |")
    print("|***************************************************|")
    
    main()