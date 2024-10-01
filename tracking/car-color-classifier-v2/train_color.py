import sys
import os
from types import SimpleNamespace
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
from torchvision import models
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time

from reid.datasets import Veri776, VeriWild
from reid.datasets import Transformations
from misc.utils import get_imagedata_info, read_image, weights_init_classifier

# import weights for EfficientNet B3/B5
from torchvision.models.efficientnet import EfficientNet_B3_Weights
from torchvision.models.efficientnet import EfficientNet_B5_Weights

class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (list): List of tuples containing the image path, folder, car_id, cam_id, model_id, color_id, type_id, timestamp.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[str, torch.Tensor, int, int, int, int, int, str]:
        img_path, folder, car_id, cam_id, model_id, color_id, type_id, timestamp = self.data[idx]

        # Read image in PIL format
        img = read_image(img_path)

        # Apply the transform if it exists
        if self.transform is not None:
            img = self.transform(img)

        index = 0

        return img_path, img, folder, index, car_id, cam_id, model_id, color_id, type_id, timestamp

    def train_collate_fn(self, batch) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, str]:
        img_paths, imgs, folders, indices, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps = zip(*batch)

        # Transform Car IDs and Images to Tensors
        # [batch_size, 3, height, width]
        imgs = torch.stack(imgs, dim=0)
        car_ids = torch.tensor(car_ids, dtype=torch.int64)  # [batch_size]
        cam_ids = torch.tensor(cam_ids, dtype=torch.int64)
        color_ids = torch.tensor(color_ids, dtype=torch.int64)

        return img_paths, imgs, folders, indices, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps

    def val_collate_fn(self, batch) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, str]:
        img_paths, imgs, folders, indices, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps = zip(*batch)

        # Transform Car IDs and Images to Tensors
        imgs = torch.stack(imgs, dim=0)                     # [batch_size, 3, height, width]
        car_ids = torch.tensor(car_ids, dtype=torch.int64)  # [batch_size]
        cam_ids = torch.tensor(cam_ids, dtype=torch.int64)
        color_ids = torch.tensor(color_ids, dtype=torch.int64)

        return img_paths, imgs, folders, indices, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps

# Variables
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_name = 'EfficientNet-B3' # 'EfficientNet-B3' or 'EfficientNet-B5'
batch_size = 16
lr = 1e-3 * (batch_size / 32)
epochs = 30
num_workers = 8
data_path = './data'
dataset_size = 100 # This means that we'll take only the 50% of the dataset

augmentation_configs = {
    'HEIGHT': 320,
    'WIDTH': 320,
    'RANDOM_HORIZONTAL_FLIP_PROB': 0.5,
    'RANDOM_CROP': (320, 320),
    'RANDOM_ERASING_PROB': 0.0,
    'JITTER_BRIGHTNESS': 0.2,
    'JITTER_CONTRAST': 0.15,
    'JITTER_SATURATION': 0.0,
    'JITTER_HUE': 0.5,
    'COLOR_AUGMENTATION': True,
    'PADDING': 0.0,
    'NORMALIZE_MEAN': None,
    'NORMALIZE_STD': None
}

# Convert the dictionary to a SimpleNamespace
augmentation_configs = SimpleNamespace(**augmentation_configs)

# For veri_776 dataset, there is not a specific dataset type
# For veri_wild, vehicle_id, and vru there are three types: small, medium, large
print("Building VeRi-776 Dataset...")
dataset_veri776 = Veri776(data_path, use_rptm=False)
print("Building VeRi-Wild Dataset...")
dataset_veriwild = VeriWild(data_path, dataset_size='small', use_rptm=False)

# Here we should relabel the classes to be the same for both datasets
# We can do this by creating a new dataset that combines both datasets
# and then relabel the classes to be the same
v776_color = dataset_veri776.color_dict = {k: v for k, v in enumerate(dataset_veri776.color_dict.values())}
vw_color = dataset_veriwild.color_dict = {k: v for k, v in enumerate(dataset_veriwild.color_dict.values())}

# Combine the color dictionaries and remove duplicates
combined_color_dict = {**v776_color}
start_index = max(combined_color_dict.keys()) + 1

for color in vw_color.values():
    if color not in combined_color_dict.values():
        combined_color_dict[start_index] = color
        start_index += 1

# Create reverse mapping for color names to indices
reverse_color_dict = {v: k for k, v in combined_color_dict.items()}

# Loop through both datasets and relabel classes
pbar = tqdm(total=sum(len(dataset.train) + len(dataset.query) + len(dataset.gallery) for dataset in [dataset_veri776, dataset_veriwild]),
            desc="Relabeling classes")

lenv776_t = len(dataset_veri776.train)
lenv776_q = len(dataset_veri776.query)
lenv776_g = len(dataset_veri776.gallery)
lenvw_t = len(dataset_veriwild.train)
lenvw_q = len(dataset_veriwild.query)
lenvw_g = len(dataset_veriwild.gallery)

for dataset in [dataset_veri776, dataset_veriwild]:
    all_data = dataset.train + dataset.query + dataset.gallery

    # Create new lists for train, query, and gallery after filtering
    filtered_train, filtered_query, filtered_gallery = [], [], []

    for idx, data in enumerate(all_data):
        if data[5] == -1:
            # Skip entries with color_id -1
            continue

        # Convert tuple to list to allow modification
        data_list = list(data)
        # Relabel color_id using reverse_color_dict
        data_list[5] = reverse_color_dict[dataset.color_dict[data[5]]]
        updated_data = tuple(data_list)

        # Add back the modified tuple to the correct dataset split
        if idx < len(dataset.train):
            filtered_train.append(updated_data)
        elif idx < len(dataset.train) + len(dataset.query):
            filtered_query.append(updated_data)
        else:
            filtered_gallery.append(updated_data)
        
        pbar.update(1)

    # Update dataset after filtering and relabeling
    dataset.train, dataset.query, dataset.gallery = filtered_train, filtered_query, filtered_gallery

pbar.close()

# Print the percentage of removed entries
lenv776_t_new = 100 - (len(dataset_veri776.train) / lenv776_t * 100)
lenv776_q_new = 100 - (len(dataset_veri776.query) / lenv776_q * 100)
lenv776_g_new = 100 - (len(dataset_veri776.gallery) / lenv776_g * 100)
lenvw_t_new = 100 - (len(dataset_veriwild.train) / lenvw_t * 100)
lenvw_q_new = 100 - (len(dataset_veriwild.query) / lenvw_q * 100)
lenvw_g_new = 100 - (len(dataset_veriwild.gallery) / lenvw_g * 100)

print("\n")
print("Filtering colors results:")
print(f"\t- Veri-776 Dataset: {lenv776_t_new + lenv776_q_new + lenv776_g_new:.3f}% of entries removed")
print(f"\t- Veri-Wild Dataset: {lenvw_t_new + lenvw_q_new + lenvw_g_new:.3f}% of entries removed")

# Transformations for the dataset
transforms = Transformations(configs=augmentation_configs)

# Reduce the dataset size if needed
if dataset_size != None:
    dataset_veri776.train = dataset_veri776.train[:int(len(dataset_veri776.train) * (dataset_size / 100))]
    dataset_veri776.query = dataset_veri776.query[:int(len(dataset_veri776.query) * (dataset_size / 100))]
    dataset_veri776.gallery = dataset_veri776.gallery[:int(len(dataset_veri776.gallery) * (dataset_size / 100))]
    dataset_veriwild.train = dataset_veriwild.train[:int(len(dataset_veriwild.train) * (dataset_size / 100))]
    dataset_veriwild.query = dataset_veriwild.query[:int(len(dataset_veriwild.query) * (dataset_size / 100))]
    dataset_veriwild.gallery = dataset_veriwild.gallery[:int(len(dataset_veriwild.gallery) * (dataset_size / 100))]
    print(f"Dataset size filtered at {dataset_size}%")

# Create the train and validation datasets
train_set = ImageDataset(data=dataset_veri776.train + dataset_veriwild.train,
                         transform=transforms.get_train_transform(),)
validation_set = ImageDataset(data=dataset_veri776.query + dataset_veri776.gallery + dataset_veriwild.query + dataset_veriwild.gallery,
                              transform=transforms.get_val_transform(),)

verbose = True
if verbose:
    num_train_pids, num_train_imgs, num_train_cams = get_imagedata_info(dataset_veri776.train + dataset_veriwild.train)
    num_query_pids, num_query_imgs, num_query_cams = get_imagedata_info(dataset_veri776.query + dataset_veriwild.query)
    num_gallery_pids, num_gallery_imgs, num_gallery_cams = get_imagedata_info(dataset_veri776.gallery + dataset_veriwild.gallery)

    print('Image Dataset statistics:')
    print('/------------------------------------------\\')
    print('|  Subset  |  # IDs | # Images | # Cameras |')
    print('|------------------------------------------|')
    print('|  Train   |  {:5d} | {:8d} | {:9d} |'.format(num_train_pids, num_train_imgs, num_train_cams))
    print('|  Query   |  {:5d} | {:8d} | {:9d} |'.format(num_query_pids, num_query_imgs, num_query_cams))
    print('|  Gallery |  {:5d} | {:8d} | {:9d} |'.format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
    print('\\------------------------------------------/')

train_loader = DataLoader(train_set, batch_size=batch_size,
                            shuffle=True,
                            collate_fn=train_set.train_collate_fn,
                            num_workers=num_workers,
                            pin_memory=False)
val_loader = DataLoader(validation_set, batch_size=batch_size,
                            shuffle=True,
                            collate_fn=validation_set.val_collate_fn,
                            num_workers=num_workers,
                            pin_memory=True)

# Load pre-trained EfficientNet model and modify the classifier
if (model_name == 'EfficientNet-B3'):
    model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
elif (model_name == 'EfficientNet-B5'):
    model = models.efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
else:
    raise ValueError(f"Model {model_name} not supported")

for param in model.parameters():
    param.requires_grad = True

# Modify classifier to match the number of classes
linear_layer = nn.Linear(in_features=model.classifier[1].in_features,
                            out_features=len(reverse_color_dict),
                            bias=True)
linear_layer.apply(weights_init_classifier)

model.classifier[1] = linear_layer

# Convert model to CUDA
model = model.to(device=device)

# Define optimizer and loss function
optimizer = Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Training loop
best_loss = float('inf')
best_acc = 0.0

for epoch in range(epochs):
    start_time = time.time()
    
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0

    # img_path, img, folders, indices, car_id, cam_id, model_id, color_id, type_id, timestamp
    for batch_idx, (_, img, _, _, _, _, _, color_id, _, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs} - Training", leave=False):
        img, color_id = img.to(device), color_id.to(device)
        
        # Forward pass
        yhat = model(img)
        loss = criterion(yhat, color_id)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        train_correct += (yhat.argmax(dim=1) == color_id).sum().item()
    
    # Average training loss and accuracy
    train_loss /= len(train_loader)
    train_acc = 100 * train_correct / len(train_loader.dataset)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    
    with torch.no_grad():
        for batch_idx, (_, img, _, _, _, _, _, color_id, _, _) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch + 1}/{epochs} - Validation", leave=False):
            img, color_id = img.to(device), color_id.to(device)
                
            # Forward pass
            yhat = model(img)
            loss = criterion(yhat, color_id)
            
            # Statistics
            val_loss += loss.item()
            val_correct += (yhat.argmax(dim=1) == color_id).sum().item()
    
    # Average validation loss and accuracy
    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / len(val_loader.dataset)
    
    # Save the best model
    if val_loss < best_loss and val_acc > best_acc:
        best_loss = val_loss
        best_acc = val_acc
        torch.save(model, f'{model_name}_loss-{val_loss:.4f}_acc-{val_acc:.4f}.pt')
        print(f"Model saved at epoch {epoch + 1} with validation loss @{val_loss:.4f} and accuracy @{val_acc:.4f}")
    
    # Print epoch results
    duration = time.time() - start_time
    print(f"Epoch {epoch + 1}/{epochs} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
          f"Time: {duration:.2f}s")