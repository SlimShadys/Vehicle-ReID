import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import _C as cfg_file
from db.database import Database
from misc.printer import Logger
from misc.utils import decompress_img, is_incomplete

# Logger
logger = Logger()

# Configs
db_cfg = cfg_file.DB

# DB connection and insertion
reid_DB = Database(db_configs=db_cfg, logger=logger)

# Query for vehicle with ID 166
myquery = {"vehicle_id": 166}

mydoc = reid_DB.bboxes_col.find(myquery, {
    "_id": 0,
    "confidence": 1,
    "bounding_box": 1,
    "cropped_image": 1,
    "timestamp": 1,
    "features": 1,
    "shape": 1,
    "vehicle_id": 1
})

# # Frame dimensions
# frame_width, frame_height = 1280, 720
# background_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
# min_box_size = 30

# # Fixed margin and central region
# margin = 100
# cv2.rectangle(background_frame, (margin, margin), (frame_width - margin, frame_height - margin), (255, 255, 255), 2)

# center_threshold = 0.30
# center_x_min = int(frame_width * (center_threshold / 2))
# center_x_max = int(frame_width * (1 - center_threshold / 2))
# center_y_min = int(frame_height * (center_threshold / 2))
# center_y_max = int(frame_height * (1 - center_threshold / 2))

# # Draw central region
# cv2.rectangle(background_frame, (center_x_min, center_y_min), (center_x_max, center_y_max), (255, 222, 173), 2)

# # Process bounding boxes
# features_array = []
# for x in tqdm(mydoc, desc="Processing Bounding Boxes"):
#     bbox = np.floor(x["bounding_box"]).astype(int)
#     cropped_image = decompress_img(x["cropped_image"], x["shape"], np.uint8)
    
#     # Resize the cropped image to the bounding box size
#     box_width, box_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
#     resized_car_image = cv2.resize(cropped_image, (box_width, box_height))
#     resized_car_image = cv2.cvtColor(resized_car_image, cv2.COLOR_BGR2RGB)
    
#     # Overlay the resized car image on the background frame
#     background_frame[bbox[1]:bbox[1]+box_height, bbox[0]:bbox[0]+box_width] = resized_car_image
    
#     # Draw the bounding box and center point
#     if is_incomplete(bbox, min_box_size, frame_width, frame_height, margin, center_threshold):
#         color = (255, 0, 0)
#     else:
#         color = (124, 252, 0)
#     cv2.rectangle(background_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

#     center_x, center_y = np.floor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]).astype(int)
#     cv2.circle(background_frame, (center_x, center_y), 5, (0, 0, 255), -1)

# # Plot the final frame with bounding boxes and car images
# plt.figure(figsize=(10, 6))
# plt.imshow(background_frame)
# plt.title(f'Vehicle {x["vehicle_id"]} Trajectory Over Time')
# plt.show()