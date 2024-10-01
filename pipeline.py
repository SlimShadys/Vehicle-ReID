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
# - Run the ReID model on the remaining frames                                         

# fare if per scegliere se fare validate color con efficientnet oppure mobilenet


import argparse

import cv2
import torch
from config import _C as cfg_file
from misc.utils import compute_iou, set_seed
from reid.datasets.transforms import Transformations
from reid.model import ModelBuilder
from tracking.model import load_yolo

# Number of classes in each dataset (Only for ID classification tasks, hence only training set is considered)
num_classes = {
    'veri_776': 576,
    'veri_wild': 30671,
    'vehicle_id': 13164,
    'vru': 7086,
}

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
    
    # Device Configuration
    device = misc_cfg.DEVICE
    if device == 'cuda' and torch.cuda.is_available():
        device = device + ":" + str(misc_cfg.GPU_ID)
    else:
        device = 'cpu'
        
    # Set seed
    seed = misc_cfg.SEED
    set_seed(seed)

    # ========================== LOAD YOLO MODEL ==========================
    yolo_model = load_yolo(tracking_cfg)
    input_path = tracking_cfg.VIDEO_PATH
    tracked_classes = tracking_cfg.MODEL.TRACKED_CLASS
    conf_thresh = tracking_cfg.MODEL.CONFIDENCE_THRESHOLD
    
    # ========================== LOAD REID MODEL ==========================
    model_name = reid_cfg.MODEL.NAME
    pretrained = reid_cfg.MODEL.PRETRAINED
    dataset_name = reid_cfg.DATASET.DATASET_NAME
    model_configs = reid_cfg.MODEL
    color_model_configs = reid_cfg.COLOR_MODEL
    model_val_path = reid_cfg.TEST.MODEL_VAL_PATH

    print("--------------------")
    print(f"Building {reid_cfg.MODEL.NAME} model...")
    model_builder = ModelBuilder(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes[dataset_name],
        model_configs=model_configs
    )
    
    # Get the model and move it to the device
    model = model_builder.move_to(device)

    # Load parameters from a .pth file
    if (model_val_path != ""):
        print("Loading model parameters from file...")
        if ('ibn' in model_name):
            model.load_param(model_val_path)
        else:
            checkpoint = torch.load(model_val_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
        print(f"Successfully loaded model parameters from file: {model_val_path}")

    model.eval()
    print(model)
    
    # ========================== LOAD COLOR MODEL ==========================
    print(f"Building color model...")
    model_builder = ModelBuilder(
        model_name='color_model',
        model_configs=color_model_configs
    )
    color_model = model_builder.move_to(device)
    print("Successfully built color model")
    
    # ======================================================================
    
    # Transformations for the dataset
    transforms = Transformations(dataset=None, configs=reid_cfg.AUGMENTATION)
    
    # Get the video loaded
    cap = cv2.VideoCapture(R"{}".format(input_path))
    video_FPS = cap.get(cv2.CAP_PROP_FPS) # Get the frame rate (FPS) of the video

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

        # Run YOLO tracking on the frame
        results = yolo_model.track(img, classes=list([idx for idx, name in tracked_classes]), conf=conf_thresh, persist=True)
        
        # Extract bounding boxes, classes, names, and confidences
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        if(results[0].boxes.id is not None):
            ids = results[0].boxes.id.tolist()
        else:
            ids = [0] * len(classes)
        names = results[0].names
        confidences = results[0].boxes.conf.tolist()

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

                full_dict[id].append({
                    'class': name,
                    'confidence': confidence,
                    'bounding_box': box,
                    'cropped_image': crop,
                    'timestamp': timestamp
                })

    # Here the cap is not opened anymore
    cap.release()
    cv2.destroyAllWindows()
    
    # Here we should filter the full_dict to keep only the bounding boxes that are coherent with the color requested
    # Run inference on the color model | color_model.predict(img)
    for id in full_dict:
        # Instead of going through all the frames, we can take all the frames at once and compute a batch prediction
        frames = [frame['cropped_image'] for frame in full_dict[id]]
        
        # Run the color model on the batch of frames
        color_predictions = color_model.color_classifier.predict(frames)
        
        for frame, color_predictions in zip(full_dict[id], color_predictions):
            color_prediction = color_predictions[0]['color'].lower()
                   
            # If this color is different from the one we want, discard the frame from the Dict
            if color_prediction not in misc_cfg.TARGET_COLOR:
                full_dict[id].remove(frame)
    
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