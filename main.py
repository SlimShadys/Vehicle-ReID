import matplotlib.pyplot as plt
import torch
from dataset import DatasetBuilder
from model import ModelBuilder
from torch.utils.data import DataLoader
from training import *
from utils import *

def main(config_path):
    # Load configuration from JSON file
    config = load_config(config_path)

    # Create Dataset
    num_classes = {
        'veri-776': 776,
        'veri-wild': 40671,
        'vehicle-id': 26267,
    }
    dataset_builder = DatasetBuilder(data_path=config['data_path'], dataset_name=config['dataset_name'])
    train_dataset = dataset_builder.train_set       # Get the Train dataset
    val_dataset = dataset_builder.validation_set   # Get the Test dataset
    print("--------------------")
    print(f"Dataset successfully built!")
    print("--------------------")

    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # # Get a simple batch from train loader and print the image
    # print("\nVisualizing a batch of images...")
    # _, img, car_id, cam_id, _, _ = next(iter(train_loader))

    # # Convert to numpy and transpose dimensions
    # img_np = img[0].permute(1, 2, 0).numpy()  # Shape: [256, 128, 3]
    # # Display the image
    # plt.figure(figsize=(8, 8))
    # plt.title(f"CarID: {car_id[0].item()}, CamID: {cam_id[0].item()}")
    # plt.imshow(img_np)
    # plt.axis('off')  # Hide axes
    # plt.show()

    print("--------------------")
    print(f"Building {config['model']} model...")
    model_builder = ModelBuilder(
        model_name=config['model'],
        pretrained=config['pretrained'],
        num_classes=num_classes[config['dataset_name']]
    )

    # Get the model and move it to the device
    model = model_builder.model
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    model = model_builder.move_to(device)

    # Print model summary
    print("--------------------")
    print(f"Model successfully built!")
    print("--------------------")
    print(f"Model Summary:")
    print(f"\t- Architecture: {config['model']}")
    print(f"\t- Number of classes: {num_classes[config['dataset_name']]}")
    print(f"\t- Pre-trained: {config['pretrained']}")
    print(f'\t- Model device: {device}')
    print(f"\t- Trainable parameters: {model_builder.get_number_trainable_parameters():,}")
    print("--------------------")

    # Create the Trainer
    trainer = Trainer(
        model=model,
        dataloaders={'train': train_loader, 'val': val_loader},
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
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
