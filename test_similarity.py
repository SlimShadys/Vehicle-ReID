import os

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from misc.utils import euclidean_dist, read_image
from reid.datasets.transforms import Transformations

def compute_multiple_images_similarity(model, device, similarity_method, val_transform):
    directory = os.path.join('data', 'test')
    images = []
    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(directory, filename)
            img = read_image(img_path)
            img_tensor = val_transform(img)
            images.append(img_tensor)

    images = torch.stack(images).to(device)
    
    with torch.no_grad():
        embeddings = model(images).squeeze().to(device)
        
    n = len(embeddings)
    similarity_matrix = torch.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):  # Only compute upper triangle
            
            # Calculate the similarity between the embeddings
            if similarity_method == 'euclidean':
                similarity = euclidean_dist(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0), train=False)
            elif similarity_method == 'cosine':
                similarity = torch.nn.functional.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).mean().item()
            
            # Since Matrix is symmetric, we can set both values at the same time
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    # Get image names
    image_names = [f for f in os.listdir(directory) if f.endswith((".png", ".jpg", ".jpeg"))]
    
    # Convert tensor to numpy array if it's not already
    if torch.is_tensor(similarity_matrix):
        similarity_matrix = similarity_matrix.cpu().numpy()

    # Create a mask to hide the upper triangle
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(similarity_matrix, 
                mask=mask,
                cmap="YlGnBu",
                vmin=-1, vmax=1,
                annot=True,
                fmt=".2f",
                square=True,
                xticklabels=image_names,
                yticklabels=image_names)

    plt.title("Image Similarity Matrix")
    plt.tight_layout()
    plt.show()
    exit(1)

from config import _C as cfg_file
from misc.printer import Logger
from misc.utils import set_seed
from reid.model import ModelBuilder
from tracking.model import load_yolo

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

# ========================== COMPUTE SIMILARITY ==========================
# Transformations needed for the ReID model
transforms = Transformations(dataset=None, configs=reid_cfg.AUGMENTATION)
val_transform = transforms.get_val_transform()

compute_multiple_images_similarity(model, device, "cosine", val_transform)