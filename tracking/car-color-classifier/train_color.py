import os
import random
import sys
from types import SimpleNamespace

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import models
# import weights for EfficientNet B3/B5
from torchvision.models.efficientnet import (EfficientNet_B3_Weights,
                                             EfficientNet_B5_Weights)
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from config import _C as cfg_file
from misc.printer import Logger
from misc.utils import get_imagedata_info, read_image, weights_init_classifier
from reid.dataset import pad_tensor_centered
from reid.datasets import Transformations, Veri776, VeriWild
from reid.model import ModelBuilder

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

        # Get shapes of all images
        shapes = [(img.shape[1], img.shape[2]) for img in imgs]
        
        # Check if all shapes are the same
        if len(set(shapes)) != 1:
            # Images have different shapes, need padding
            max_h = max(shape[0] for shape in shapes)
            max_w = max(shape[1] for shape in shapes)
            
            padded_imgs = [pad_tensor_centered(img, (max_h, max_w)) for img in imgs]
            imgs = padded_imgs

        # Transform Car IDs and Images to Tensors
        imgs = torch.stack(imgs, dim=0) # [batch_size, 3, height, width]
        car_ids = torch.tensor(car_ids, dtype=torch.int64)  # [batch_size]
        cam_ids = torch.tensor(cam_ids, dtype=torch.int64)
        color_ids = torch.tensor(color_ids, dtype=torch.int64)

        return img_paths, imgs, folders, indices, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps

    def val_collate_fn(self, batch) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, str]:
        img_paths, imgs, folders, indices, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps = zip(*batch)

        # Get shapes of all images
        shapes = [(img.shape[1], img.shape[2]) for img in imgs]
        
        # Check if all shapes are the same
        if len(set(shapes)) != 1:
            # Images have different shapes, need padding
            max_h = max(shape[0] for shape in shapes)
            max_w = max(shape[1] for shape in shapes)
            
            padded_imgs = [pad_tensor_centered(img, (max_h, max_w)) for img in imgs]
            imgs = padded_imgs

        # Alternative: Simply resize the images to a fixed size, in this case (320, 320)
        # imgs = torch.stack([torch.nn.functional.interpolate(
        #                         img.unsqueeze(0), size=(320, 320),
        #                         mode='bilinear', align_corners=False).squeeze(0) 
        #                     for img in imgs], dim=0)

        # Transform Car IDs and Images to Tensors
        imgs = torch.stack(imgs, dim=0)                     # [batch_size, 3, height, width]
        car_ids = torch.tensor(car_ids, dtype=torch.int64)  # [batch_size]
        cam_ids = torch.tensor(cam_ids, dtype=torch.int64)
        color_ids = torch.tensor(color_ids, dtype=torch.int64)

        return img_paths, imgs, folders, indices, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps

def main():
    # Various configs
    misc_cfg = cfg_file.MISC
    reid_cfg = cfg_file.REID
    color_model_configs = reid_cfg.COLOR_MODEL
    
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

    model_name = color_model_configs.NAME # 'EfficientNet-B3' / 'EfficientNet-B5' / 'svm'
    batch_size = reid_cfg.TRAINING.BATCH_SIZE
    lr = 1e-3 * (batch_size / 32)
    epochs = reid_cfg.TRAINING.EPOCHS
    data_path = reid_cfg.DATASET.DATA_PATH
    dataset_size = 100 # This means that we'll take only 100% of the dataset

    # Augmentation configurations
    augmentation_configs = {
        'RESIZE': 320,
        'RANDOM_CROP': 320,
        'RANDOM_HORIZONTAL_FLIP_PROB': 0.5,
        'RANDOM_ERASING_PROB': 0.0,
        'JITTER_BRIGHTNESS': 0.0,
        'JITTER_CONTRAST': 0.0,
        'JITTER_SATURATION': 0.0,
        'JITTER_HUE': 0.0,
        'COLOR_AUGMENTATION': False,
        'PADDING': 0,
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

    print(reverse_color_dict)

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

    # Reduce the dataset size if needed
    if dataset_size != None:
        dataset_veri776.train = dataset_veri776.train[:int(len(dataset_veri776.train) * (dataset_size / 100))]
        dataset_veri776.query = dataset_veri776.query[:int(len(dataset_veri776.query) * (dataset_size / 100))]
        dataset_veri776.gallery = dataset_veri776.gallery[:int(len(dataset_veri776.gallery) * (dataset_size / 100))]
        dataset_veriwild.train = dataset_veriwild.train[:int(len(dataset_veriwild.train) * (dataset_size / 100))]
        dataset_veriwild.query = dataset_veriwild.query[:int(len(dataset_veriwild.query) * (dataset_size / 100))]
        dataset_veriwild.gallery = dataset_veriwild.gallery[:int(len(dataset_veriwild.gallery) * (dataset_size / 100))]
        print(f"Dataset size filtered at {dataset_size}%")

    # Transformations for the dataset
    transforms = Transformations(configs=augmentation_configs)

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
                                num_workers=0,
                                pin_memory=False)
    val_loader = DataLoader(validation_set, batch_size=batch_size,
                                shuffle=True,
                                collate_fn=validation_set.val_collate_fn,
                                num_workers=0,
                                pin_memory=True)

    # Number of classes in each dataset (Only for ID classification tasks, hence only training set is considered)
    num_classes = {
        'veri_776': 576,
        'veri_wild': 30671,
        'vehicle_id': 13164,
        'vru': 7086,
    }

    # Load Logger
    logger = Logger()

    if model_name == 'svm':
        import pickle

        # ========================== LOAD REID MODEL ========================== #
        model_name = reid_cfg.MODEL.NAME
        pretrained = reid_cfg.MODEL.PRETRAINED
        dataset_name = reid_cfg.DATASET.DATASET_NAME
        model_configs = reid_cfg.MODEL
        model_val_path = reid_cfg.TEST.MODEL_VAL_PATH
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

        # Extract features for training and validation sets
        train_features = []
        train_labels = []
        with torch.no_grad():  # No need to compute gradients for feature extraction
            for batch_idx, (_, imgs, _, _, car_ids, _, _, color_id, _, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Extracting Training Features", leave=False):
                imgs = imgs.to(device)
                output = model(imgs)  # Extract features
                output = output.view(output.size(0), -1)  # Flatten the features
                train_features.append(output.cpu().numpy())
                train_labels.append(color_id.cpu().numpy())

        train_features = np.vstack(train_features)      # Combine all batches into a single array
        train_labels = np.concatenate(train_labels)     # Combine all batches of labels

        # Extract features for training and validation sets
        val_features = []
        val_labels = []
        with torch.no_grad():  # No need to compute gradients for feature extraction
            for batch_idx, (_, imgs, _, _, car_ids, _, _, color_id, _, _) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Extracting Val Features", leave=False):
                imgs = imgs.to(device)
                output = model(imgs)  # Extract features
                output = output.view(output.size(0), -1)  # Flatten the features
                val_features.append(output.cpu().numpy())
                val_labels.append(color_id.cpu().numpy())

        val_features = np.vstack(val_features)      # Combine all batches into a single array
        val_labels = np.concatenate(val_labels)     # Combine all batches of labels

        # Encode the labels (car IDs) as integers
        le = LabelEncoder()
        train_labels = le.fit_transform(train_labels)
        val_labels = le.transform(val_labels)

        # Train the SVM classifier
        svm_classifier = SVC(kernel='linear')
        print("Training SVM Classifier...")
        svm_classifier.fit(train_features, train_labels)

        # Evaluate on the validation set
        val_predictions = svm_classifier.predict(val_features)
        accuracy = accuracy_score(val_labels, val_predictions)

        print(f"SVM Classifier Accuracy: {accuracy * 100:.2f}%")

        # Save the SVM model to a file
        with open('svm_classifier.pkl', 'wb') as f:
            pickle.dump(svm_classifier, f)

        print("SVM Classifier saved to 'svm_classifier.pkl'")

    if model_name == 'efficientnet-b3':
        model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    elif (model_name == 'efficientnet-b5'):
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

if __name__ == '__main__':
    main()