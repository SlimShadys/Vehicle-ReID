import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random

import numpy as np
import torch
from config import _C as cfg_file
from reid.dataset import (BalancedIdentitySampler, CombinedDataset, DatasetBuilder,
                     RandomIdentitySampler)
from reid.loss import LossBuilder
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

def main(config, seed):

    # Set the seed for reproducibility
    set_seed(seed)

    # ============= VARIABLES =============
    misc_configs = config.MISC
    dataset_configs = config.REID.DATASET
    model_configs = config.REID.MODEL
    augmentation_configs = config.REID.AUGMENTATION
    loss_configs = config.REID.LOSS
    training_configs = config.REID.TRAINING
    val_configs = config.REID.VALIDATION
    test_configs = config.REID.TEST
    
    # Misc variables
    use_amp = misc_configs.USE_AMP
    output_dir = misc_configs.OUTPUT_DIR
    device = misc_configs.DEVICE
    
    # Dataset variables
    data_path = dataset_configs.DATA_PATH
    dataset_name = dataset_configs.DATASET_NAME
    split_perc = dataset_configs.SPLITTINGS
    dataset_size = dataset_configs.DATASET_SIZE
    sampler_type = dataset_configs.SAMPLER_TYPE
    num_instances = dataset_configs.NUM_INSTANCES

    # Model parameters
    model_name = model_configs.NAME
    pretrained = model_configs.PRETRAINED
    fine_tuning = model_configs.RUN_FINETUNING
    model_configs.PADDING_MODE = augmentation_configs.PADDING_MODE

    # Loss parameters
    loss_type = loss_configs.TYPE
    alpha = loss_configs.ALPHA
    k = loss_configs.K
    margin = loss_configs.MARGIN
    label_smoothing = loss_configs.LABEL_SMOOTHING
    apply_MALW = loss_configs.APPLY_MALW
    use_rptm, _ = loss_configs.USE_RPTM

    # Training parameters
    batch_size = training_configs.BATCH_SIZE
    num_workers = training_configs.NUM_WORKERS

    # Validation parameters
    batch_size_val = val_configs.BATCH_SIZE
    val_interval = val_configs.VAL_INTERVAL

    # =====================================

    print("|***************************************************|")
    print("|                   Vehicle Re-ID                   |")
    print("|---------------------------------------------------|")
    print("|             Made by: Gianmarco Scarano            |")
    print("|       --- MSc Student at AI & Robotics ---        |")
    print("|      --- University of Rome, La Sapienza ---      |")
    print("|         ---        Rome, Italy        ---         |")
    print("|***************************************************|")

    if device == 'cuda' and torch.cuda.is_available():
        print('Cuda available: {}'.format(torch.cuda.is_available()))
        if (torch.cuda.device_count() > 1):
            print(f"Detected multiple GPUs. Devices: {torch.cuda.device_count()}")
            print(f"Using the GPU number: {misc_configs.GPU_ID}")
        device = torch.device(device=device + ":" + str(misc_configs.GPU_ID))
        print("GPU: " + torch.cuda.get_device_name(device))
        print("Total memory: {:.1f} GB".format((float(torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)))))
    else:
        device = torch.device("cpu")
        print('Cuda not available, so using CPU. Please consider switching to a GPU runtime before running the script!')

    print("===================================================")
    print(F"Torch version: {torch.__version__}")
    print("===================================================")

    # # Number of classes in each dataset (Only for ID classification tasks, hence only training set is considered)
    # num_classes = {
    #     'ai_city': 440,
    #     'ai_city_mix': 1802,
    #     'ai_city_sim': 1362,
    #     'vehicle_id': 13164,
    #     'veri_776': 576,
    #     'veri_wild': 30671,
    #     'vric': 2811,
    #     'vru': 7086,
    # }

    # Create Dataset and DataLoaders
    print(f"Building Dataset:")
    print(f"- Name: {dataset_name}")
    if type(dataset_name) == str and dataset_name in ['veri_wild', 'vehicle_id', 'vru']:
        print(f"- Size: {dataset_size}")
    print("--------------------")

    # Extract the splittings
    if split_perc is not None:
        splittings = {}
        for (dataset, proportion) in split_perc:
            splittings[dataset] = proportion

    dataset_builder = DatasetBuilder(data_path=data_path, dataset_name=dataset_name,
                                     dataset_size=dataset_size, use_rptm=use_rptm,
                                     augmentation_configs=augmentation_configs,
                                     splittings=splittings)
    train_dataset = dataset_builder.train_set       # Get the Train dataset
    val_dataset = dataset_builder.validation_set    # Get the Test dataset
    num_classes = dataset_builder.dataset.get_unique_car_ids()
    print(f"Dataset successfully built!")
    print(f"Unique classes: {num_classes}")

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
    
    if val_dataset is None or type(dataset_builder.dataset) == CombinedDataset:
        val_loader = None
        print("NOTE: Validation dataset is None. Skipping validation...")
    else:
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
        num_classes=num_classes,
        model_configs=model_configs,
        device=device
    )

    # Get the model and move it to the device
    model = model_builder.move_to(device)

    # ============== FINE - TUNING ==============
    if fine_tuning:
        # Ideally, we simply must load the weights up to the last layer and re-train the last classification layer
        val_path = cfg_file.REID.TEST.MODEL_VAL_PATH
        if (val_path is not None):
            print("Loading model parameters from file...")
            if ('ibn' in cfg_file.REID.MODEL.NAME):
                try:
                    param_dict = torch.load(val_path)
                except:
                    param_dict = torch.load(val_path, map_location='cuda:0')
                for param_name in param_dict['model']:
                    if 'classifier.weight' in param_name:
                        print("FINE-TUNING: Skipping the last classification layer...")
                        continue
                    else:
                        model.state_dict()[param_name].copy_(param_dict['model'][param_name])
            print(f"Successfully loaded model parameters from file: {val_path}")
    # ==========================================

    # Print model summary
    print("--------------------")
    print(f"Model successfully built!")
    print("--------------------")
    print(f"Model Summary:")
    print(f"\t- Architecture: {model_name}")
    print(f"\t- Number of classes: {num_classes}")
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
    output_config = os.path.join(output_dir, "config.yaml")
    with open(output_config, "w") as f: f.write(cfg_file.dump())
    print(f"Config file saved to: {output_config}")

    # Create the Trainer
    trainer = Trainer(
        model=[model, None],
        val_interval=val_interval,
        dataloaders={'train': train_loader, 'val': (val_loader, len(dataset_builder.dataset.query)),
                     'dataset': dataset_builder.dataset, 'transform': dataset_builder.transforms},
        loss_fn=loss_fn,
        device=device,
        configs=(misc_configs, augmentation_configs, loss_configs, training_configs, val_configs, test_configs)
    )
    trainer.run()

# Usage: python main.py <path_to_config.yml>
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vehicle Re-ID - Train")
    parser.add_argument("--config_file", default="", help="Path to config file", type=str)
    args = parser.parse_args()

    # Load the config file
    if args.config_file != "":
        cfg_file.merge_from_file(args.config_file)
    else:
        print("No config file specified. Running with default config file from config.py!")
    
    # Get the seed from the config
    seed = cfg_file.MISC.SEED

    # Run the main function with the seed
    main(cfg_file, seed)