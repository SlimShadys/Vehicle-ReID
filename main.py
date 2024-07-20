import matplotlib.pyplot as plt
from dataset import DatasetBuilder
from model import ModelBuilder
from torch.utils.data import DataLoader
from training import *
from utils import *

def main(config_path):
    # Load configuration from JSON file
    config = load_config(config_path)

    # ============= VARIABLES =============
    # Global variables
    data_path = config['data_path']
    dataset_name = config['dataset_name']
    dataset_size = config['dataset_size']

    # Model parameters
    device = config['device']
    model = config['model']
    pretrained = config['pretrained']
    
    # Training parameters
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    # =====================================

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Get a simple batch from train loader and print the image
    print("\nVisualizing a batch of images...")
    img_path, img, car_id, cam_id, model_id, color_id, type_id, timestamp = next(iter(train_loader))

    # Convert to numpy and transpose dimensions
    img_np = img[0].permute(1, 2, 0).numpy()  # Shape: [320, 320, 3]
    
    # Display the image
    plt.figure(figsize=(8, 8))
    plt.title(f"CarID: {car_id[0].item()}, CamID: {cam_id[0].item()} | Timestamp: {timestamp[0]}")
    plt.imshow(img_np)
    plt.axis('off') # Hide axes
    plt.show()

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

    # Create the Trainer
    trainer = Trainer(
        model=model,
        dataloaders={'train': train_loader, 'val': val_loader},
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )
    trainer.train()

if __name__ == '__main__':
    # import sys
    # if len(sys.argv) != 2:
    #     print("Usage: python script.py <path_to_config.json>")
    #     sys.exit(1)
    # main(sys.argv[1])

    config_file = 'config.json'

    main(config_file)
