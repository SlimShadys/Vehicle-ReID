import matplotlib.pyplot as plt
from dataset import DatasetBuilder
from model import ModelBuilder
from torch.utils.data import DataLoader
from training import *
from utils import normalize, read_image
import torchvision.transforms as transforms

import yaml
import os

def main(config):

    # ============= VARIABLES =============
    dataset_config = config['dataset']
    model_config = config['model']
    val_config = config['validation']
    test_config = config['test']

    # Dataset variables
    data_path = dataset_config['data_path']
    dataset_name = dataset_config['dataset_name']
    dataset_size = dataset_config['dataset_size']

    # Model parameters
    device = model_config['device']
    model = model_config['model']
    pretrained = model_config['pretrained']
    model_val_path = model_config['model_val_path']
    
    # Validation parameters
    batch_size_val = val_config['batch_size']
    
    # Test parameters
    run_reid_metrics = test_config['run_reid_metrics']
    normalize_embeddings = test_config['normalize_embeddings']
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
        num_classes=num_classes[dataset_name]
    )

    # Get the model and move it to the device
    model = model_builder.move_to(device)

    # Load parameters from a .pth file
    model.load_state_dict(torch.load(model_val_path))
    model.eval()
    print(model)
    print("Successfully loaded model parameters from file: ", model_val_path)

    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),          # Resize to 384x384
        transforms.ToTensor(),                  # Convert to tensor
    ])

    if not run_reid_metrics:
        # Read 2 images from /data/test
        path_img_1 = os.path.join("data", "test", "honda_f.jpg")
        path_img_2 = os.path.join("data", "test", "matiz_f.jpg")
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
        distance = torch.nn.functional.pairwise_distance(features_1, features_2)
        print(f"Distance between the 2 images: {distance.item()}")
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

        # Create the Trainer
        trainer = Trainer(
            model=model,
            val_interval=0,
            dataloaders={'train': None, 'val': {val_loader, len(dataset_builder.dataset.query)}},
            loss_fn=None,
            epochs=None,
            learning_rate=None,
            device=device
        )
        trainer.validate(epoch=0, save_results=False)
    
if __name__ == '__main__':
    # import sys
    # if len(sys.argv) != 2:
    #     print("Usage: python script.py <path_to_config.json>")
    #     sys.exit(1)
    # main(sys.argv[1])

    # Parameters from config.yml file
    with open('config.yml', 'r') as f:
        config = yaml.load(f, yaml.FullLoader)['reid']

    main(config)
