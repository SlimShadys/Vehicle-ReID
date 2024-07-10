import glob
import os
import re
import xml.etree.ElementTree as ET

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils import read_image

class DatasetBuilder():
    def __init__(self, data_path, dataset_name):
        """
        Args:
            data_path (string): Path to the data file.
            dataset_name (string): Name of the dataset.
        """
        self.data_path = data_path
        self.dataset_name = dataset_name

        if self.dataset_name == 'veri-776':
            self.dataset = Veri776(self.data_path)
        elif self.dataset_name == 'veri-wild':
            self.dataset = VeriWild()
        elif self.dataset_name == 'vehicle-id':
            self.dataset = VehicleID()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        self.train_set = ImageDataset(self.dataset.train, transform=self.dataset.train_transform)
        self.validation_set = ImageDataset(self.dataset.gallery + self.dataset.query, transform=self.dataset.val_transform)

class Veri776():
    def __init__(self, data_path):
        # Generic variables
        self.data_path = os.path.join(data_path, 'VeRi-776')
        self.train_labels = self.get_labels(os.path.join(self.data_path, 'train_label.xml'))
        self.test_labels = self.get_labels(os.path.join(self.data_path, 'test_label.xml'))

        # Directories
        self.train_dir = os.path.join(self.data_path,'image_train')
        self.query_dir = os.path.join(self.data_path, 'image_query')
        self.gallery_dir = os.path.join(self.data_path, 'image_test')

        # Train
        self.train = self.get_list(self.train_labels, self.train_dir)

        # Test (query and gallery)
        self.gallery = self.get_list(self.test_labels, self.gallery_dir)
        self.query = self.get_list(None, self.query_dir) # No details for the query set

        # Junk index and GT index
        self.indices = self.get_indices('name_query.txt', 'jk_index.txt', 'gt_index.txt')

        # Specific transforms for Veri-776
        self.train_transform = transforms.Compose([
            transforms.Resize((320, 320)),          # Resize to 320x320
            transforms.RandomCrop((320, 320)),      # Random crop to 320x320
            transforms.RandomHorizontalFlip(p=0.5), # Random horizontal flip
            transforms.Pad(10),                     # Pad the image with 10 pixels
            transforms.ToTensor(),                  # Convert to tensor
        ])

        # Specific transforms for Veri-776
        self.val_transform = transforms.Compose([
            transforms.Resize((320, 320)),          # Resize to 320x320
            transforms.ToTensor(),                  # Convert to tensor
        ])

    def get_indices(self, query_file_path, junk_file_path, gt_file_path):
        # Read query file names
        with open(os.path.join(self.data_path, query_file_path), 'r') as file:
            query_names = [line.strip() for line in file.readlines()]

        # Read junk indices
        with open(os.path.join(self.data_path, junk_file_path), 'r') as file:
            junk_indices = [line.strip().split() for line in file.readlines()]

        # Read ground truth indices
        with open(os.path.join(self.data_path, gt_file_path), 'r') as file:
            gt_indices = [line.strip().split() for line in file.readlines()]

        # Construct the dictionary
        index_dict = {}
        for i, query_name in enumerate(query_names):
            index_dict[query_name] = {
                'junk': list(map(int, junk_indices[i])),
                'gt': list(map(int, gt_indices[i]))
            }

        return index_dict

    def get_labels(self, file_path):
        # Open the file with the correct encoding
        with open(file_path, 'r', encoding='gb2312') as file:
            xml_content = file.read()
        
        # Parse the XML string
        root = ET.fromstring(xml_content)
        
        # Dictionary to store all items
        labels_dict = {}
        
        # Iterate through each Item element
        for item in root.findall('.//Item'):
            # Extract numeric part of cameraID
            camera_id_str = item.get('cameraID')
            vehicle_id_str = item.get('vehicleID')
            camera_id_num = int(re.search(r'\d+', camera_id_str).group())
            car_id_num = int(re.search(r'\d+', vehicle_id_str).group())

            item_info = {
                'vehicle_ID': car_id_num,
                'camera_ID': camera_id_num,
                'color_ID': int(item.get('colorID')),
                'type_ID': int(item.get('typeID'))
            }
            labels_dict[item.get('imageName')] = item_info
        
        return labels_dict

    def get_list(self, labels, dir_path):
        if 'train' in dir_path or 'test' in dir_path:
            add_details = True
        else:
            add_details = False
        dataset = []
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))

        for img_path in img_paths:
            if(add_details):
                try:
                    # Details
                    car_detail = labels[os.path.basename(img_path)]
                    car_id = car_detail['vehicle_ID']
                    cam_id = car_detail['camera_ID']
                    color_id = car_detail['color_ID']
                    type_id = car_detail['type_ID']
                    details = (img_path, car_id, cam_id, color_id, type_id)
                except:
                    # There are 32 images that are not in the XML file,
                    # So we need to extract the infos manually from the image name
                    details = self.retrieve_details(img_path)
            else:
                details = self.retrieve_details(img_path)

            dataset.append(details)

        return dataset

    def retrieve_details(self, img_path):
        # We need to extract the infos manually from the image name
        # such as: '0002_c004_00084250_0.jpg'
        #
        # This should happen only when we are dealing with the query set
        # or, in general, if we want to recover the details from the image name
        img_full_string = os.path.basename(img_path).split('_')

        # Extract the vehicle and camera IDs
        car_id_num = int(re.search(r'\d+', img_full_string[0]).group())
        camera_id_num = int(re.search(r'\d+', img_full_string[1]).group())
        
        # Details
        car_id = car_id_num
        cam_id = camera_id_num
        color_id = None
        type_id = None

        return (img_path, car_id, cam_id, color_id, type_id)

class VeriWild():
    pass

class VehicleID():
    pass

class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (numpy array): Numpy array containing the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, car_id, cam_id, color_id, type_id = self.data[idx]
        img = read_image(img_path)

        # Apply the transform if it exists
        if self.transform is not None:
            img = self.transform(img)
        
        # Maybe applying a possible collate_fn here would be a good idea
        car_id = torch.tensor(car_id, dtype=torch.int64)

        return img_path, img, car_id, cam_id, color_id, type_id

