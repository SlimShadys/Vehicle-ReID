import os
import random
import sys

import numpy as np
import torch
import yaml
from dataset import DatasetBuilder
from datasets.transforms import Transformations
from misc.utils import euclidean_dist, normalize, read_image
from model import ModelBuilder
from torch.utils.data import DataLoader
from train import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Correctly set the seed to:", seed)

def main(config, seed):

    # Set the seed for reproducibility
    set_seed(seed)

    print("|***************************************************|")
    print("|                   Vehicle Re-ID                   |")
    print("|                        TEST                       |")
    print("|---------------------------------------------------|")
    print("|             Made by: Gianmarco Scarano            |")
    print("|       --- MSc Student at AI & Robotics ---        |")
    print("|      --- University of Rome, La Sapienza ---      |")
    print("|         ---        Rome, Italy        ---         |")
    print("|***************************************************|")

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
    print(model)

    if not run_reid_metrics:
        # Read 2 images from /data/test
        path_img_1 = os.path.join("data", "test", "honda_f.jpg")
        path_img_2 = os.path.join("data", "test", "honda_f_2.jpg")
        print(F"Reading the following images:")
        print(F"- {path_img_1}")
        print(F"- {path_img_2}")

        img1 = read_image(path_img_1)
        img2 = read_image(path_img_2)

        # Define the validation transformations
        transforms = Transformations(configs=augmentation_configs)
        val_transform = transforms.get_val_transform()

        # Apply validation transformations
        img1 = val_transform(img1).unsqueeze(0).to(device)
        img2 = val_transform(img2).unsqueeze(0).to(device)

        features_1 = model(img1, training=False)
        features_2 = model(img2, training=False)

        # Normalize the embeddings
        if normalize_embeddings:
            # Normalize embeddings for the Metric Loss
            features_1 = normalize(features_1)
            # Normalize embeddings for the Metric Loss
            features_2 = normalize(features_2)

        # Calculate the L2 distance between the 2 images
        # Smaller distances indicate higher similarity, while larger distances indicate less similarity
        distance = euclidean_dist(features_1, features_2, train=False)
        print(f"Distance between the 2 images: {distance.item():.4f}")
    else:
        # Create Dataset and DataLoaders
        print(f"Building Dataset:")
        print(f"- Name: {dataset_name}")
        if dataset_name in ['veri_wild', 'vehicle_id']:
            print(f"- Size: {dataset_size}")
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

        # Create the Trainer
        trainer = Trainer(
            model=model,
            val_interval=0,
            dataloaders={'train': None,
                         'val': {val_loader, len(dataset_builder.dataset.query)},
                         'dataset': dataset_builder.dataset,
                         'transform': dataset_builder.transforms},
            loss_fn=None,
            device=device,
            configs=(misc_configs, None, None, None, val_configs, test_configs)
        )
        trainer.validate(save_results=False)

# Usage: python test.py <path_to_config.yml>
if __name__ == '__main__':
    config_file = 'config_test.yml'
    if len(sys.argv) != 2:
        print("You might be using an IDE to run the script or forgot to append the config file. Running with default config file: 'config.yml'")
    else:
        config_file = sys.argv[1]

    # Parameters from config.yml file
    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)['reid']

    # Get the seed from the config
    # Default to 2047315 if not specified
    seed = config.get('misc', {}).get('seed', 2047315)

    # Run the main function with the seed
    main(config, seed)