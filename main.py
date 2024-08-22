import os
import random
import shutil
import sys

import numpy as np
import torch
import yaml
from dataset import (BalancedIdentitySampler, DatasetBuilder,
                     RandomIdentitySampler)
from loss import LossBuilder
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

def main(config, config_file, seed):

    # Set the seed for reproducibility
    set_seed(seed)

    # ============= VARIABLES =============
    misc_configs = config['misc']
    dataset_configs = config['dataset']
    model_configs = config['model']
    augmentation_configs = config['augmentation']
    loss_configs = config['loss']
    training_configs = config['training']
    val_configs = config['validation']
    test_configs = config['test']

    # Misc variables
    use_amp = misc_configs['use_amp']
    output_dir = misc_configs['output_dir']
    
    # Dataset variables
    data_path = dataset_configs['data_path']
    dataset_name = dataset_configs['dataset_name']
    dataset_size = dataset_configs['dataset_size']
    sampler_type = dataset_configs['sampler_type']
    num_instances = dataset_configs['num_instances']

    # Model parameters
    device = model_configs['device']
    model_name = model_configs['model']
    pretrained = model_configs['pretrained']

    # Loss parameters
    loss_type = loss_configs['type']
    alpha = loss_configs['alpha']
    k = loss_configs['k']
    margin = loss_configs['margin']
    label_smoothing = loss_configs['label_smoothing']
    apply_MALW = loss_configs['apply_MALW']
    use_rptm, _ = loss_configs['use_rptm']

    # Training parameters
    batch_size = training_configs['batch_size']
    num_workers = training_configs['num_workers']

    # Validation parameters
    batch_size_val = val_configs['batch_size']
    val_interval = val_configs['val_interval']

    # =====================================

    print("|***************************************************|")
    print("|                   Vehicle Re-ID                   |")
    print("|---------------------------------------------------|")
    print("|             Made by: Gianmarco Scarano            |")
    print("|       --- MSc Student at AI & Robotics ---        |")
    print("|      --- University of Rome, La Sapienza ---      |")
    print("|         ---        Rome, Italy        ---         |")
    print("|***************************************************|")

    if ('cuda' in device and torch.cuda.is_available()):
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
        'veri_776': 576,
        'veri_wild': 30671,
        'vehicle_id': 13164,
        'vru': 7086,
    }

    # Create Dataset and DataLoaders
    print(f"Building Dataset:")
    print(f"- Name: {dataset_name}")
    if dataset_name in ['veri_wild', 'vehicle_id', 'vru']:
        print(f"- Size: {dataset_size}")
    print("--------------------")

    dataset_builder = DatasetBuilder(data_path=data_path, dataset_name=dataset_name,
                                     dataset_size=dataset_size, use_rptm=use_rptm,
                                     augmentation_configs=augmentation_configs)
    train_dataset = dataset_builder.train_set       # Get the Train dataset
    val_dataset = dataset_builder.validation_set    # Get the Test dataset
    print(f"Dataset successfully built!")
    print(f"Unique classes: {dataset_builder.dataset.get_unique_car_ids()}")

    # Create the DataLoaders
    if sampler_type == 'random':
        sampler = RandomIdentitySampler(dataset_builder.dataset.train, batch_size=batch_size, num_instances=num_instances)
    elif sampler_type == 'balanced':
        sampler = BalancedIdentitySampler(dataset_builder.dataset.train, batch_size=batch_size, num_instances=num_instances)
    else:
        sampler = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler,
                              shuffle=True if sampler is None else None,
                              collate_fn=train_dataset.train_collate_fn,
                              num_workers=num_workers,
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val,
                            shuffle=False,
                            collate_fn=val_dataset.val_collate_fn,
                            num_workers=num_workers,
                            pin_memory=True, drop_last=False)

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

    # Print model summary
    print("--------------------")
    print(f"Model successfully built!")
    print("--------------------")
    print(f"Model Summary:")
    print(f"\t- Architecture: {model_name}")
    print(f"\t- Number of classes: {num_classes[dataset_name]}")
    print(f"\t- Pre-trained: {pretrained}")
    print(f'\t- Model device: {device}')
    print(f"\t- Trainable parameters: {model_builder.get_number_trainable_parameters():,}")
    print("--------------------")

    # Define the loss function
    loss_fn = LossBuilder(loss_type=loss_type, alpha=alpha, k=k, margin=margin,
                          label_smoothing=label_smoothing, apply_MALW=apply_MALW,
                          batch_size=batch_size, use_amp=use_amp)

    # Create the output directory and save the config file
    print("Creating the output directory...")
    if (os.path.exists(output_dir) == False):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")

    # Save the config file
    output_config = os.path.join(output_dir, config_file)
    shutil.copy(config_file, output_config)
    print(f"Config file saved to: {output_config}")

    # Create the Trainer
    trainer = Trainer(
        model=model,
        val_interval=val_interval,
        dataloaders={'train': train_loader, 'val': {val_loader, len(dataset_builder.dataset.query)},
                     'dataset': dataset_builder.dataset, 'transform': dataset_builder.transforms},
        loss_fn=loss_fn,
        device=device,
        configs=(misc_configs, augmentation_configs, loss_configs, training_configs, val_configs, test_configs)
    )
    trainer.run()

# Usage: python main.py <path_to_config.yml>
if __name__ == '__main__':
    config_file = 'config.yml'
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
    main(config, config_file, seed)