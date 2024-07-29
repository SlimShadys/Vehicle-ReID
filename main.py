import matplotlib.pyplot as plt
from dataset import DatasetBuilder
from model import ModelBuilder
from loss import LossBuilder
from torch.utils.data import DataLoader
from training import *
from utils import *
from dataset import RandomIdentitySampler

import yaml

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
    batch_size = training_config['batch_size']
    
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

    if(torch.cuda.is_available()):
        print('Cuda available: {}'.format(torch.cuda.is_available()))
        if (torch.cuda.device_count() > 1):
            print(f"Detected multiple GPUs. Devices: {torch.cuda.device_count()}")
            print("Using the first GPU available...")
        device = torch.device("cuda:0")
        
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
                              num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val,
                            shuffle=False,
                            collate_fn=val_dataset.val_collate_fn,
                            num_workers=4)

    # # Get a simple batch from train loader and print the image
    # print("\nVisualizing a batch of images...")
    # img_path, img, car_id, cam_id, model_id, color_id, type_id, timestamp = next(iter(train_loader))

    # # Convert to numpy and transpose dimensions
    # img_np = img[0].permute(1, 2, 0).numpy()  # Shape: [320, 320, 3]
    
    # # Display the image
    # plt.figure(figsize=(8, 8))
    # plt.title(f"CarID: {car_id[0].item()}, CamID: {cam_id[0].item()} | Timestamp: {timestamp[0]}")
    # plt.imshow(img_np)
    # plt.axis('off') # Hide axes
    # plt.show()

    print("--------------------")
    print(f"Building {model} model...")
    model_builder = ModelBuilder(
        model_name=model,
        pretrained=pretrained,
        num_classes=num_classes[dataset_name]
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
    loss_fn = LossBuilder(alpha=alpha, k=k, margin=margin, label_smoothing=label_smoothing, apply_MALW=apply_MALW, num_classes=num_classes[dataset_name])

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
