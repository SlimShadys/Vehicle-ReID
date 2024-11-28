import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import random
import sys
from config import _C as cfg_file

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from reid.dataset import DatasetBuilder
from reid.datasets.transforms import Transformations
from misc.utils import euclidean_dist, read_image
from reid.model import ModelBuilder
from torch.utils.data import DataLoader
from reid.training.trainer import Trainer

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

def compute_multiple_images_similarity(model, device, similarity_method, normalize_embeddings, val_transform):
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
    misc_configs = config.MISC
    dataset_configs = config.REID.DATASET
    model_configs = config.REID.MODEL
    color_configs = config.REID.COLOR_MODEL
    augmentation_configs = config.REID.AUGMENTATION
    val_configs = config.REID.VALIDATION
    test_configs = config.REID.TEST

    # Misc variables
    device = misc_configs.DEVICE

    # Dataset variables
    data_path = dataset_configs.DATA_PATH
    dataset_name = dataset_configs.DATASET_NAME
    dataset_size = dataset_configs.DATASET_SIZE

    # Model parameters
    model_name = model_configs.NAME
    pretrained = model_configs.PRETRAINED

    # Validation parameters
    batch_size_val = val_configs.BATCH_SIZE

    # Test parameters
    run_reid_metrics = test_configs.RUN_REID_METRICS
    run_color_metrics = test_configs.RUN_COLOR_METRICS
    stack_images = test_configs.STACK_IMAGES
    similarity = test_configs.SIMILARITY
    normalize_embeddings = test_configs.NORMALIZE_EMBEDDINGS
    model_val_path = test_configs.MODEL_VAL_PATH
    img_path_1 = test_configs.PATH_IMG_1
    img_path_2 = test_configs.PATH_IMG_2
    # =====================================

    # Number of classes in each dataset (Only for ID classification tasks, hence only training set is considered)
    num_classes = {
        'ai_city': 440,
        'ai_city_sim': 1802,
        'vehicle_id': 13164,
        'veri_776': 576,
        'veri_wild': 30671,
        'vric': 2811,
        'vru': 7086,
    }
    
    model = None

    if run_reid_metrics == True:
        print("--------------------")
        print(f"Building {model_name} model...")
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
        if dataset_name in ['vehicle_id']:
            raise ValueError(f"Unfortunately this dataset doesn't have color infos for Test vehicles.")
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
            # Build the color model
            print(f"Loading color model from {color_configs.NAME}...")
            model_builder = ModelBuilder(
                model_name=color_configs.NAME,
                pretrained=pretrained,
                num_classes=num_classes[dataset_name],
                model_configs=color_configs,
                device=device
            )
            color_model = model_builder.move_to(device)
            print(f"Successfully loaded color model from {color_configs.NAME}!")

            if ('svm' in color_configs.NAME):
                print(f"Building {model_name} model for SVM Color Recognition...")
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
                if (model_val_path is not None):
                    print("Loading model parameters from file...")
                    if ('ibn' in model_name):
                        model.load_param(model_val_path)
                    else:
                        checkpoint = torch.load(model_val_path, map_location=device)
                        model.load_state_dict(checkpoint['model'])
                    print(f"Successfully loaded model parameters from file: {model_val_path}")

                model.eval()    
        else:
            color_model = None

        # Create the Trainer
        trainer = Trainer(
            model=[model, color_model],
            val_interval=0,
            dataloaders={'train': None, 'val': (val_loader, len(dataset_builder.dataset.query)),
                        'dataset': dataset_builder.dataset, 'transform': dataset_builder.transforms},
            loss_fn=None,
            device=device,
            configs=(misc_configs, None, None, None, val_configs, test_configs)
        )
        if run_reid_metrics: metrics.append('reid')
        if run_color_metrics: metrics.append('color')
        
        trainer.validate(save_results=False, metrics=metrics)

# Usage: python main.py <path_to_config.yml>
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vehicle Re-ID - Test")
    parser.add_argument("--config_file", default="configs/config_test.yml", help="Path to config file", type=str)
    args = parser.parse_args()

    # Load the config file
    if args.config_file != "":
        cfg_file.merge_from_file(args.config_file)
    else:
        print("No config file specified. Running with default config file!")
    
    # Get the seed from the config
    seed = cfg_file.MISC.SEED

    # Run the main function with the seed
    main(cfg_file, seed)