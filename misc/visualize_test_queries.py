import argparse
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from reid.dataset import DatasetBuilder
from reid.model import ModelBuilder
from torch.utils.data import DataLoader

# ============= FUNCTIONS =============
def get_scores(query_feature, gallery_features):
    query = query_feature.view(-1, 1)
    score = torch.mm(gallery_features, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    return score

def show_query_result(axes, query_img, gallery_imgs, query_label, gallery_labels):
    query_trans = transforms.Pad(4, 0)
    good_trans = transforms.Pad(4, (0, 255, 0))
    bad_trans = transforms.Pad(4, (255, 0, 0))

    for idx, img in enumerate([query_img] + gallery_imgs):
        img = img.resize((128, 128))
        if idx == 0:
            img = query_trans(img)
        elif query_label == gallery_labels[idx - 1]:
            img = good_trans(img)
        else:
            img = bad_trans(img)

        ax = axes.flat[idx]
        ax.imshow(img)

    for i in range(len(axes.flat)):
        ax = axes.flat[i]
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
        ax.axis("off")

def on_key(event):
    """If a left or right key was pressed, plots the next query in that direction."""
    global curr_idx
    if event.key == "left":
        curr_idx = (curr_idx - 1) if curr_idx > 0 else len(queries) - 1
    elif event.key == "right":
        curr_idx = (curr_idx + 1) if curr_idx < len(queries) - 1 else 0
    elif event.key == "enter":
        fig.savefig(os.path.join('misc', 'reid_query_result.pdf'), pad_inches=0, bbox_inches='tight')
    elif event.key == "escape": # If 'ESC' is pressed, close the plot
        plt.close()
    else:
        return
    refresh_plot()

def get_image(dataset, idx):
    """Returns the image at a given index of the dataset."""
    if dataset == 'query':
        img_path = dataset_builder.dataset.query[idx][0]
    else:
        img_path = dataset_builder.dataset.gallery[idx][0]
    image = Image.open(img_path).convert("RGB")
    return image

def refresh_plot():
    """Computes the result of the current query and shows it on the canvas."""
    q_feature = query_features[curr_idx]
    y = query_labels[curr_idx]

    if use_cam:
        curr_cam = dataset_builder.dataset.query[curr_idx][3] # Because the 4th element is the camera ID
        cam_ids = torch.tensor([tup[3] for tup in dataset_builder.dataset.gallery])
        good_gallery_idx = cam_ids != curr_cam
        gallery_orig_idx = np.where(good_gallery_idx)[0]
        gal_features = gallery_features[good_gallery_idx]
    else:
        gallery_orig_idx = np.arange(len(dataset_builder.dataset.gallery))
        gal_features = gallery_features
    gallery_scores = get_scores(q_feature, gal_features)
    idx = np.argsort(gallery_scores)[::-1]

    if use_cam:
        g_labels = gallery_labels[gallery_orig_idx.copy()][idx.copy()]
    else:
        g_labels = gallery_labels[idx.copy()]

    q_img = get_image('query', curr_idx)
    g_imgs = [get_image('gallery', gallery_orig_idx[i])
              for i in idx[:args.num_images]]
    show_query_result(axes, q_img, g_imgs, y, g_labels)
    fig.canvas.draw()
    fig.canvas.flush_events()

######################################################################
# Options
# =============
config_file = 'config_test.yml'

# Parameters from config.yml file
with open(config_file, 'r') as f:
    config = yaml.load(f, yaml.FullLoader)['reid']

# Get the seed from the config
# Default to 2047315 if not specified
seed = config.get('misc', {}).get('seed', 2047315)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("Correctly set the seed to:", seed)

parser = argparse.ArgumentParser(description="Show sample queries and retrieved gallery images for a reid model")
parser.add_argument("--input_size", type=int, default=320, help="Image input size for the model")
parser.add_argument("--num_images", type=int, default=10, help="number of gallery images to show")
parser.add_argument("--imgs_per_row", type=int, default=5)
parser.add_argument("--use_saved_mat", type=bool, default=True, help="Use precomputed features from a previous test.py run: pytorch_result.mat.")
parser.add_argument("--use_cam", type=bool, default=True, help="Use camera information to filter out gallery images.")
args = parser.parse_args()
use_cam = args.use_cam

# ============= VARIABLES =============
misc_configs = config['misc']
dataset_configs = config['dataset']
model_configs = config['model']
augmentation_configs = config['augmentation']
val_configs = config['validation']
test_configs = config['test']

# Dataset variables
data_path = dataset_configs['data_path']
dataset_name = dataset_configs['dataset_name']
dataset_size = dataset_configs['dataset_size']

# Model parameters
device = model_configs['device']
model_name = model_configs['model']
pretrained = model_configs['pretrained']

# Validation parameters
batch_size_val = val_configs['batch_size']

# Test parameters
run_reid_metrics = test_configs['run_reid_metrics']
normalize_embeddings = test_configs['normalize_embeddings']
model_val_path = test_configs['model_val_path']
# =====================================

######################################################################
# Load Data
# ---------
#
print(f"Building Dataset:")
print(f"- Name: {dataset_name}")
print("--------------------")

dataset_builder = DatasetBuilder(data_path=data_path, dataset_name=dataset_name,
                                    dataset_size=dataset_size,
                                    augmentation_configs=augmentation_configs)
val_dataset = dataset_builder.validation_set   # Get the Test dataset
print(f"Dataset successfully built!")
print(f"Unique classes: {dataset_builder.dataset.get_unique_car_ids()}")

# Create the DataLoaders
val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False,
                        collate_fn=val_dataset.val_collate_fn, num_workers=0,
                        pin_memory=True, drop_last=False)

######################################################################
# Run queries
# -----------
#
if args.use_saved_mat:
    #name_mat = 'pytorch_result_{}_{}.mat'.format(model_name, model_val_path.split(os.sep)[-2])
    name_mat = 'pytorch_result_veri776_resnet50_ibn_a_Test-14.mat'
    saved_res = scipy.io.loadmat(os.path.join('misc', name_mat))
    gallery_features = torch.Tensor(saved_res["gallery_features"])
    gallery_labels = saved_res["gallery_label"].reshape(-1)
    query_features = torch.Tensor(saved_res["query_features"])
    query_labels = saved_res["query_label"].reshape(-1)
else:
    num_classes = {
        'veri_776': 576,
        'veri_wild': 30671,
        'vehicle_id': 13164,
    }

    print("--------------------")
    print(f"Building {model_name} model...")
    model_builder = ModelBuilder(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes[dataset_name],
        model_configs=model_configs
    )

    # Get the model and move it to the device
    model = model_builder.move_to(device)

    # Load parameters from a .pth file
    if (model_val_path is not None):
        if ('ibn' in model_name):
            model.load_param(model_val_path)
        else:
            checkpoint = torch.load(model_val_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
        print(f"Successfully loaded model parameters from file: {model_val_path}")

    model.eval()

    print("Computing gallery features ...")
    num_query = len(dataset_builder.dataset.query)
    features, car_ids, cam_ids, paths = [], [], [], []

    with torch.no_grad():
        for i, (img_path, img, folders, indices, car_id, cam_id, model_id, color_id, type_id, timestamp) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Running Validation"):
            img, car_id, cam_id = img.to(device), car_id.to(device), cam_id.to(device)

            with torch.no_grad():
                feat = model(img, training=False).detach().cpu()

            features.append(feat)
            car_ids.append(car_id)
            cam_ids.append(cam_id)
            paths.extend(img_path)

    features = torch.cat(features, dim=0)
    car_ids = torch.cat(car_ids, dim=0)
    cam_ids = torch.cat(cam_ids, dim=0)

    query_features = features[:num_query]
    query_labels = car_ids[:num_query]

    gallery_features = features[num_query:]
    gallery_labels = car_ids[num_query:]
    
    # After computing the features, car_ids, and cam_ids
    result = {
        'gallery_features': gallery_features.cpu().numpy(),
        'gallery_label': gallery_labels.cpu().numpy(),
        'query_features': query_features.cpu().numpy(),
        'query_label': query_labels.cpu().numpy(),
    }
    filename = 'pytorch_result_{}_{}.mat'.format(model_name, model_val_path.split(os.sep)[-2])
    scipy.io.savemat(os.path.join('misc', filename), result)
    
dataset = dataset_builder.dataset.query
queries = list(range(len(dataset)))
random.shuffle(queries)

n_rows = math.ceil((1 + args.num_images) / args.imgs_per_row)
fig, axes = plt.subplots(n_rows, args.imgs_per_row, figsize=(12, 15))
fig.canvas.mpl_connect('key_press_event', on_key)

print("- Press <left-arrow> and <right-arrow> to navigate queries.\n"
      "- Press <enter> to save into current folder as pdf.\n"
      "- Press <esc> to close the plot.")

curr_idx = 0
refresh_plot()
plt.show()