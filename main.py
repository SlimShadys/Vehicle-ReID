import os
import shutil
import sys

import matplotlib.pyplot as plt
import torch
import yaml
from dataset import DatasetBuilder, RandomIdentitySampler
from loss import LossBuilder
from model import ModelBuilder
from torch.utils.data import DataLoader
from train import Trainer

def main(config):

    # ============= VARIABLES =============
    dataset_config = config['dataset']
    model_config = config['model']
    training_config = config['training']
    val_config = config['validation']
    loss_config = config['loss']

    # Dataset variables
    data_path = dataset_config['data_path']
    dataset_name = dataset_config['dataset_name']
    dataset_size = dataset_config['dataset_size']

    # Model parameters
    device = model_config['device']
    model = model_config['model']
    pretrained = model_config['pretrained']
    
    # Training parameters
    output_dir = training_config['output_dir']
    batch_size = training_config['batch_size']
    num_workers = training_config['num_workers']
    
    # Validation parameters
    batch_size_val = val_config['batch_size']
    val_interval = val_config['val_interval']

    # Loss parameters
    alpha = loss_config['alpha']
    k = loss_config['k']
    margin = loss_config['margin']
    label_smoothing = loss_config['label_smoothing']
    apply_MALW = loss_config['apply_MALW']
    # =====================================

    print("|***************************************************|")
    print("|                   Vehicle Re-ID                   |")
    print("|---------------------------------------------------|")
    print("|             Made by: Gianmarco Scarano            |")
    print("|       --- MSc Student at AI & Robotics ---        |")
    print("|      --- University of Rome, La Sapienza ---      |")
    print("|         ---        Rome, Italy        ---         |")
    print("|***************************************************|")

    if(torch.cuda.is_available()):
        print('Cuda available: {}'.format(torch.cuda.is_available()))
        if (torch.cuda.device_count() > 1):
            print(f"Detected multiple GPUs. Devices: {torch.cuda.device_count()}")
            print(f"Using the GPU number: {device.split(':')[1]}")
        device = torch.device(device=device)
        
        print("GPU: " + torch.cuda.get_device_name(device))
        print("Total memory: {:.1f} GB".format((float(torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)))))
    else:
        device = torch.device("cpu")
        print('Cuda not available, so using CPU. Please consider switching to a GPU runtime before running the notebook!')
        
    print("===================================================")
    print(F"Torch version: {torch.__version__}")
    print("===================================================")

    # Number of classes in each dataset (Only for ID classification tasks, hence only training set is considered)
    num_classes = {
        'veri-776': 576,
        'veri-wild': 30671,
        'vehicle-id': 13164,
    }

    # Create Dataset and DataLoaders
    print(f"Building Dataset:")
    print(f"- Name: {dataset_name}")
    if dataset_name in ['veri-wild', 'vehicle-id']:
        print(f"- Size: {dataset_size}")
    print("--------------------")

    dataset_builder = DatasetBuilder(data_path=data_path, dataset_name=dataset_name, dataset_size=dataset_size)
    train_dataset = dataset_builder.train_set       # Get the Train dataset
    val_dataset = dataset_builder.validation_set   # Get the Test dataset
    print(f"Dataset successfully built!")
    print(f"Unique classes: {dataset_builder.dataset.get_unique_car_ids()}")

    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=RandomIdentitySampler(dataset_builder.dataset.train, batch_size=batch_size, num_instances=6),
                              collate_fn=train_dataset.train_collate_fn,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val,
                            shuffle=False,
                            collate_fn=val_dataset.val_collate_fn,
                            num_workers=num_workers)
    
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
   
    # Print model summary
    print("--------------------")
    print(f"Model successfully built!")
    print("--------------------")
    print(f"Model Summary:")
    print(f"\t- Architecture: {model}")
    print(f"\t- Number of classes: {num_classes[dataset_name]}")
    print(f"\t- Pre-trained: {pretrained}")
    print(f'\t- Model device: {device}')
    print(f"\t- Trainable parameters: {model_builder.get_number_trainable_parameters():,}")
    print("--------------------")

    # Define the loss function
    loss_fn = LossBuilder(alpha=alpha, k=k, margin=margin, label_smoothing=label_smoothing, apply_MALW=apply_MALW)

    # Create the output directory and save the config file
    print("Creating the output directory...")
    if(os.path.exists(output_dir) == False):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")
    
    # Save the config file
    output_config = os.path.join(output_dir, 'config.yml')
    shutil.copy('config.yml', output_config)
    print(f"Config file saved to: {output_config}")
    
    # Create the Trainer
    trainer = Trainer(
        model=model,
        val_interval=val_interval,
        dataloaders={'train': train_loader, 'val': {val_loader, len(dataset_builder.dataset.query)}},
        loss_fn=loss_fn,
        device=device,
        train_configs=training_config
    )
    trainer.run()

# Usage: python main.py <path_to_config.yml>
if __name__ == '__main__':
    config_file = 'config.yml'
    if len(sys.argv) != 2:
        print("You might be using an IDE to run the script. Running with default config file: 'config.yml'")
    else:
        config_file = sys.argv[1]
        
    # Parameters from config.yml file
    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)['reid']
    
    # Run the main function
    main(config)