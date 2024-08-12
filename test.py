import os
import sys

import torch
import yaml
from dataset import DatasetBuilder
from datasets.transforms import Transformations
from misc.utils import euclidean_dist, normalize, read_image
from model import ModelBuilder
from torch.utils.data import DataLoader
from train import Trainer

def main(config):

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
    dataset_config = config['dataset']
    model_config = config['model']
    val_config = config['validation']
    test_config = config['test']
    training_configs = config['training']
    
    # Dataset variables
    data_path = dataset_config['data_path']
    dataset_name = dataset_config['dataset_name']
    dataset_size = dataset_config['dataset_size']

    # Model parameters
    device = model_config['device']
    model = model_config['model']
    pretrained = model_config['pretrained']
        
    # Validation parameters
    batch_size_val = val_config['batch_size']
    
    # Test parameters
    run_reid_metrics = test_config['run_reid_metrics']
    normalize_embeddings = test_config['normalize_embeddings']
    model_val_path = test_config['model_val_path']
    # =====================================

    num_classes = {
        'veri-776': 576,
        'veri-wild': 30671,
        'vehicle-id': 13164,
    }

    print("--------------------")
    print(f"Building {model} model...")
    model_builder = ModelBuilder(
        model_name=model,
        pretrained=pretrained,
        num_classes=num_classes[dataset_name],
        model_config=model_config
    )

    # Get the model and move it to the device
    model = model_builder.move_to(device)

    # Load parameters from a .pth file
    if(model_val_path is not None):
        checkpoint = torch.load(model_val_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"Successfully loaded model parameters from file: {model_val_path}")

    model.eval()
    print(model)

    # Define the validation transformations
    transforms = Transformations()
    val_transform = transforms.get_val_transform()

    if not run_reid_metrics:
        # Read 2 images from /data/test
        path_img_1 = os.path.join("data", "test", "matiz_b.jpg")
        path_img_2 = os.path.join("data", "test", "matiz_b_2.jpg")
        print(F"Reading the following images:")
        print(F"- {path_img_1}")
        print(F"- {path_img_2}")

        img1 = read_image(path_img_1)
        img2 = read_image(path_img_2)

        # Apply validation transformations
        img1 = val_transform(img1).unsqueeze(0).to(device)
        img2 = val_transform(img2).unsqueeze(0).to(device)

        features_1 = model(img1, training=False)
        features_2 = model(img2, training=False)
        
        # Normalize the embeddings
        if normalize_embeddings:
            features_1 = normalize(features_1) # Normalize embeddings for the Metric Loss
            features_2 = normalize(features_2) # Normalize embeddings for the Metric Loss

        # Calculate the L2 distance between the 2 images
        # Smaller distances indicate higher similarity, while larger distances indicate less similarity
        distance = euclidean_dist(features_1, features_2)
        print(f"Distance between the 2 images: {distance.item():.4f}")
    else:
        # Create Dataset and DataLoaders
        print(f"Building Dataset:")
        print(f"- Name: {dataset_name}")
        if dataset_name in ['veri-wild', 'vehicle-id']:
            print(f"- Size: {dataset_size}")
        print("--------------------")

        dataset_builder = DatasetBuilder(data_path=data_path, dataset_name=dataset_name, dataset_size=dataset_size)
        val_dataset = dataset_builder.validation_set   # Get the Test dataset
        print(f"Dataset successfully built!")
        print(f"Unique classes: {dataset_builder.dataset.get_unique_car_ids()}")

        # Create the DataLoaders
        val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, collate_fn=val_dataset.val_collate_fn, num_workers=0)

        # Remove the learning rate from the training configs, so we do not initialize the optimizer and scheduler
        training_configs['learning_rate'] = None 

        # Create the Trainer
        trainer = Trainer(
            model=model,
            val_interval=0,
            dataloaders={'train': None, 'val': {val_loader, len(dataset_builder.dataset.query)}},
            loss_fn=None,
            device=device,
            train_configs=training_configs,
            val_configs=val_config,
        )
        trainer.validate(save_results=False)
    
# Usage: python test.py <path_to_config.yml>
if __name__ == '__main__':
    config_file = 'config.yml'
    if len(sys.argv) != 2:
        print("You might be using an IDE to run the script or forgot to append the config file. Running with default config file: 'config.yml'")
    else:
        config_file = sys.argv[1]
        
    # Parameters from config.yml file
    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)['reid']
    
    # Run the main function
    main(config)