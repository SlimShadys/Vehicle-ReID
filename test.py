import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from dataset import DatasetBuilder
from datasets.transforms import Transformations
from misc.utils import euclidean_dist, read_image
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

def compute_two_images_similarity(model, device, similarity, normalize_embeddings, val_transform, paths):
        # Read 2 images from the config_test.yml file
        path_img_1, path_img_2 = paths
        
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
            print("Normalizing!")
            # Normalize embeddings for the Metric Loss
            features_1 = torch.nn.functional.normalize(features_1, p=2, dim=1)
            features_2 = torch.nn.functional.normalize(features_2, p=2, dim=1)

        # Calculate the L2 distance between the 2 images
        # Smaller distances indicate higher similarity, while larger distances indicate less similarity
        if similarity == 'euclidean':
            distance = euclidean_dist(features_1, features_2, train=False)
        # Calculate the Cosine similarity between the 2 images
        # Larger values indicate higher similarity, while smaller values indicate less similarity
        elif similarity == 'cosine':
            distance = torch.nn.functional.cosine_similarity(features_1, features_2).mean()
        else:
            raise ValueError(f"Unsupported similarity metric: {similarity}")

        return distance

def compute_multiple_images_similarity(model, device, similarity, normalize_embeddings, val_transform):
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
            if similarity == 'euclidean':
                similarity = euclidean_dist(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0), train=False)
            elif similarity == 'cosine':
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
    run_color_metrics = test_configs['run_color_metrics']
    stack_images = test_configs['stack_images']
    similarity = test_configs['similarity']
    normalize_embeddings = test_configs['normalize_embeddings']
    model_val_path = test_configs['model_val_path']
    img_path_1 = test_configs['path_img_1']
    img_path_2 = test_configs['path_img_2']
    # =====================================

    # Number of classes in each dataset (Only for ID classification tasks, hence only training set is considered)
    num_classes = {
        'veri_776': 576,
        'veri_wild': 30671,
        'vehicle_id': 13164,
        'vru': 7086,
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
        print("Loading model parameters from file...")
        if ('ibn' in model_name):
            model.load_param(model_val_path)
        else:
            checkpoint = torch.load(model_val_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
        print(f"Successfully loaded model parameters from file: {model_val_path}")

    model.eval()
    print(model)

    if run_reid_metrics == False and run_color_metrics == False:
        # Define the validation transformations
        transforms = Transformations(configs=augmentation_configs)
        val_transform = transforms.get_val_transform()

        if stack_images == True:
            # No distance return, since it simply plots the similarity matrix which is more intuitive
            compute_multiple_images_similarity(model, device, similarity, normalize_embeddings, val_transform)
        else:
            distance = compute_two_images_similarity(model, device, similarity, normalize_embeddings, val_transform, [img_path_1, img_path_2])

            if (similarity == 'cosine'):
                print(f"Similarity between the 2 images: {distance}")
            else:
                print(f"Distance between the 2 images: {distance}")      
    elif run_reid_metrics == True or run_color_metrics == True:

        # Metrics to run
        metrics = []

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

        if run_color_metrics:
            print(f"Building color model...")
            model_builder = ModelBuilder(
                model_name='color_model',
                pretrained=pretrained,
                num_classes=num_classes[dataset_name],
                model_configs=model_configs
            )
            color_model = model_builder.move_to(device)
        else:
            color_model = None

        # Create the Trainer
        trainer = Trainer(
            model=[model, color_model],
            val_interval=0,
            dataloaders={'train': None, 'val': {val_loader, len(dataset_builder.dataset.query)},
                        'dataset': dataset_builder.dataset, 'transform': dataset_builder.transforms},
            loss_fn=None,
            device=device,
            configs=(misc_configs, None, None, None, val_configs, test_configs)
        )
        if run_reid_metrics: metrics.append('reid')
        if run_color_metrics: metrics.append('color')
        
        trainer.validate(save_results=False, metrics=metrics)

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