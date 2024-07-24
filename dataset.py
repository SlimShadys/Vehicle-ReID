import glob
import os
import re
import xml.etree.ElementTree as ET

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils import read_image

class DatasetBuilder():
    def __init__(self, data_path, dataset_name, dataset_size='small'):
        """
        Args:
            data_path (string): Path to the data file.
            dataset_name (string): Name of the dataset.
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.dataset_size = dataset_size

        # For veri-776 dataset, there is not a specific dataset type
        # For veri-wild and vehicle-id, there are three types: small, medium, large
        if self.dataset_name == 'veri-776':
            self.dataset = Veri776(self.data_path)
        elif self.dataset_name == 'veri-wild':
            self.dataset = VeriWild(self.data_path, self.dataset_size)
        elif self.dataset_name == 'vehicle-id':
            self.dataset = VehicleID(self.data_path, self.dataset_size)
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
            transforms.Resize((384, 384)),          # Resize to 384x384
            transforms.RandomCrop((384, 384)),      # Random crop to 384x384
            transforms.RandomHorizontalFlip(p=0.5), # Random horizontal flip
            transforms.Pad(10),                     # Pad the image with 10 pixels
            transforms.ToTensor(),                  # Convert to tensor
        ])

        # Specific transforms for Veri-776
        self.val_transform = transforms.Compose([
            transforms.Resize((384, 384)),          # Resize to 384x384
            transforms.ToTensor(),                  # Convert to tensor
        ])

        self.len = self.get_unique_car_ids()

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
            camera_id_num = int(re.search(r'\d+', camera_id_str).group()) - 1 # Subtract 1 to start from 0
            car_id_num = int(re.search(r'\d+', vehicle_id_str).group()) - 1 # Subtract 1 to start from 0

            item_info = {
                'vehicle_ID': car_id_num,
                'camera_ID': camera_id_num,
                'model_ID': -1, # Model ID is not available in the XML file
                'type_ID': int(item.get('typeID')),
                'color_ID': int(item.get('colorID')),
                'timestamp': 'None'
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
                    model_id = car_detail['model_ID']
                    type_id = car_detail['type_ID']
                    color_id = car_detail['color_ID']
                    timestamp = car_detail['timestamp']
                    details = (img_path, car_id, cam_id, model_id, color_id, type_id, timestamp)
                except:
                    # There are 32 images that are not in the XML file,
                    # So we need to extract the infos manually from the image name
                    details = self.retrieve_details(img_path)
            else:
                details = self.retrieve_details(img_path)

            dataset.append(details)

        # Relabel the train set only
        if 'train' in dir_path:
            dataset = self.relabel_ids(dataset)

        return dataset

    def relabel_ids(self, dataset):
        all_pids = {}
        relabeled_dataset = []
        for item in dataset:
            img_path, vehicle_id, camera_id, model_id, color_id, type_id, timestamp = item

            if vehicle_id not in all_pids:
                all_pids[vehicle_id] = len(all_pids)

            new_id = all_pids[vehicle_id]
            relabeled_dataset.append((img_path, new_id, camera_id, model_id, color_id, type_id, timestamp))
        return relabeled_dataset

    def retrieve_details(self, img_path):
        # We need to extract the infos manually from the image name
        # such as: '0002_c004_00084250_0.jpg'
        #
        # This should happen only when we are dealing with the query set
        # or, in general, if we want to recover the details from the image name
        img_full_string = os.path.basename(img_path).split('_')

        # Extract the vehicle and camera IDs
        car_id_num = int(re.search(r'\d+', img_full_string[0]).group()) - 1 # Subtract 1 to start from 0
        camera_id_num = int(re.search(r'\d+', img_full_string[1]).group()) - 1 # Subtract 1 to start from 0
        
        # Details | Color ID and Type ID are not available in the image name, so we set them to -1
        car_id = car_id_num
        cam_id = camera_id_num
        model_id = -1
        type_id = -1
        color_id = -1
        timestamp = 'None'

        return (img_path, car_id, cam_id, model_id, color_id, type_id, timestamp)

    def get_unique_car_ids(self):
        # Combine all car IDs from train, query, and gallery sets
        all_car_ids = {car_id for _, car_id, _, _, _, _, _ in self.train}
        # Get the unique car IDs
        return len(all_car_ids)

class VeriWild():
    def __init__(self, data_path, dataset_size='small'):
        # Generic variables
        self.data_path = os.path.join(data_path, 'VeRi-Wild')
        self.dataset_sizes = {'small': 3000, 'medium': 5000, 'large': 10000} # Number of test images - 3000 (Small), 5000 (Medium), 10000 (Large)
        self.num_test = self.dataset_sizes[dataset_size]

        self.vehicle_infos = self.get_vehicle_infos(os.path.join(self.data_path, 'train_test_split', 'vehicle_info.txt'))

        # Directory
        self.img_dir = os.path.join(self.data_path, 'images')

        # Train, Test and Query
        self.train = self.get_list(os.path.join(self.data_path, 'train_test_split', 'train_list_start0.txt'))
        self.gallery = self.get_list(os.path.join(self.data_path, 'train_test_split', f'test_{self.num_test}_id.txt'))
        self.query = self.get_list(os.path.join(self.data_path, 'train_test_split', f'test_{self.num_test}_id_query.txt'))

        # Relabel the train set only while keeping the original IDs for the query and gallery sets
        self.train = self.relabel_ids(self.train)

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

        self.len = self.get_unique_car_ids()

    def relabel_ids(self, dataset):
        all_pids = {}
        relabeled_dataset = []
        for item in dataset:
            img_path, vehicle_id, camera_id, model_id, color_id, type_id, timestamp = item

            if vehicle_id not in all_pids:
                all_pids[vehicle_id] = len(all_pids)

            new_id = all_pids[vehicle_id]
            relabeled_dataset.append((img_path, new_id, camera_id, model_id, color_id, type_id, timestamp))
        return relabeled_dataset

    def get_vehicle_infos(self, file_path):
        # Open the file with the correct encoding
        with open(file_path, 'r') as file:
            lines = [line.strip().split(';') for line in file.readlines()]
        
        # Dictionary to store all items
        vehicle_infos = {}

        # Before returning the final Dictionary, we need to convert the model_name, color_name, and type_name to a proper ID label
        # We can do this by creating a dictionary for each of these attributes
        # Then, we can iterate through the dictionary and replace the name with the ID
        model_dict = {}
        color_dict = {}
        type_dict = {}

        # Iterate through each Item element
        for line in lines:
            vehicle_id = int(line[0].split('/')[0])

            # If vehicle_id is not in the dictionary, create a new list
            if vehicle_id not in vehicle_infos:
                vehicle_infos[vehicle_id] = []

            item_info = {
                'vehicle_ID': vehicle_id,
                'vehicle_img': int(line[0].split('/')[1]),
                'model_name': line[3],
                'camera_ID': int(line[1]),
                'color_name': line[5],
                'type_name': line[4],
                'timestamp': line[2]
            }
            vehicle_infos[vehicle_id].append(item_info)

        # Remap model, color and type labels to proper IDs
        for vehicle_id, vehicle_info in vehicle_infos.items():
            for item in vehicle_info:
                model_name = item['model_name']
                color_name = item['color_name']
                type_name = item['type_name']

                if model_name not in model_dict:
                    model_dict[model_name] = len(model_dict)
                if color_name not in color_dict:
                    color_dict[color_name] = len(color_dict)
                if type_name not in type_dict:
                    type_dict[type_name] = len(type_dict)

                item['model_ID'] = model_dict[model_name]
                item['color_ID'] = color_dict[color_name]
                item['type_ID'] = type_dict[type_name]

        return vehicle_infos

    def get_list(self, file_path):
        # List to store all items
        dataset = []

        # Read query file names
        with open(file_path, 'r') as file:
            lines = [line.strip().split(' ') for line in file.readlines()]
                
        # Iterate through each Item element
        for line in lines:
            # Retrieve the row from self.vehicle_infos using the vehicle ID and the dictionary using the vehicle_img
            vehicle_id = int(line[0].split('/')[0])
            vehicle_img = int(line[0].split('/')[1].split('.')[0])
            vehicle_info = self.vehicle_infos[vehicle_id]

            # Extract the details
            try:
                result = next(row for row in vehicle_info if row['vehicle_img'] == vehicle_img)
            except StopIteration:
                print(f"No matching row found for vehicle_img {vehicle_img}")

            dataset.append((os.path.join(self.img_dir, line[0]),
                            result['vehicle_ID'],
                            result['camera_ID'],
                            result['model_ID'], # result['model_name'] -> If we want the string name of the model
                            result['color_ID'], # result['color_name'] -> If we want the string name of the color
                            result['type_ID'], # result['type_name'] -> If we want the string name of the type
                            result['timestamp']))
        return dataset

    def get_unique_car_ids(self):
        # Combine all car IDs from train, query, and gallery sets
        all_car_ids = {car_id for _, car_id, _, _, _, _, _ in self.train}
        # Get the unique car IDs
        return len(all_car_ids)

class VehicleID():
    def __init__(self, data_path, dataset_size='small'):
        # Generic variables
        self.data_path = os.path.join(data_path, 'VehicleID')
        self.dataset_sizes = {'small': 800, 'medium': 1600, 'large': 2400} # Number of test images - 800 (Small), 1600 (Medium), 2400 (Large)
        self.num_test = self.dataset_sizes[dataset_size]

        self.vehicle_infos = self.get_vehicle_infos(os.path.join(self.data_path, 'attribute'))

        # Directory
        self.img_dir = os.path.join(self.data_path, 'image')

        # Train, Test and Query
        self.train = self.get_list(os.path.join(self.data_path, 'train_test_split', 'train_list.txt'))
        
        # Test (query and gallery) - WE DO NOT HAVE ACCESS TO THOSE FILES YET
        # self.gallery = self.get_list(os.path.join(self.data_path, 'train_test_split', f'test_list_{self.num_test}_gallery.txt'))
        # self.query = self.get_list(os.path.join(self.data_path, 'train_test_split', f'test_{self.num_test}_query.txt'))

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

        self.len = self.get_unique_car_ids()

    def get_vehicle_infos(self, file_path):
        # Image to Vehicle ID mapping
        img2vid = {}
        with open(os.path.join(file_path, 'img2vid.txt'), 'r') as f:
            for line in f:
                img, vid = line.strip().split()
                img2vid[img] = int(vid)

        # Vehicle ID to Color ID mapping
        vid2color = {}
        with open(os.path.join(file_path, 'color_attr.txt'), 'r') as f:
            for line in f:
                vid, color_id = line.strip().split()
                vid2color[int(vid)] = int(color_id)

        # Vehicle ID to Model ID mapping
        vid2model = {}
        with open(os.path.join(file_path, 'model_attr.txt'), 'r') as f:
            for line in f:
                vid, model_id = line.strip().split()
                vid2model[int(vid)] = int(model_id)

        final_dict = {}

        for img, vid in img2vid.items():
            color_id = vid2color.get(vid, -1)  # -1 if color is not found
            model_id = vid2model.get(vid, -1)  # -1 if model is not found
            
            final_dict[img] = {
                'car_id': vid,
                'model_id': model_id,
                'color_id': color_id
            }

        return final_dict

    def get_list(self, file_path):
        # List to store all items
        dataset = []

        # Read query file names
        with open(file_path, 'r') as file:
            lines = [line.strip().split(' ') for line in file.readlines()]
                
        # Iterate through each Item element
        for line in lines:
            # Retrieve the row from self.vehicle_infos using the vehicle ID and the dictionary using the vehicle_img
            vehicle_img = line[0]
            vehicle_id = int(line[1])

            # Extract the details
            try:
                vehicle_info = self.vehicle_infos[vehicle_img]
                dataset.append((os.path.join(self.img_dir, line[0] + '.jpg',),
                                vehicle_info['car_id'],
                                -1, # Camera ID is not available
                                vehicle_info['model_id'], # model ID is the model name
                                vehicle_info['color_id'], # Color ID is the color name
                                -1, # Type ID is not available
                                -1,) # Timestamp is not available
                                ) 
            except StopIteration:
                print(f"No matching details found for Vehicle with ID: {vehicle_id}")

        return dataset

    def get_unique_car_ids(self):
        # Combine all car IDs from train, query, and gallery sets
        all_car_ids = {car_id for _, car_id, _, _, _, _, _ in self.train}
        # Get the unique car IDs
        return len(all_car_ids)

class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (numpy array): Numpy array containing the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, car_id, cam_id, model_id, color_id, type_id, timestamp = self.data[idx]

        # Read image in PIL format
        img = read_image(img_path)

        # Apply the transform if it exists
        if self.transform is not None:
            img = self.transform(img)

        return img_path, img, car_id, cam_id, model_id, color_id, type_id, timestamp

    def train_collate_fn(self, batch):
        img_paths, imgs, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps = zip(*batch)

        # Transform Car IDs and Images to Tensors
        car_ids = torch.tensor(car_ids, dtype=torch.int64)  # [batch_size]
        imgs = torch.stack(imgs, dim=0)                     # [batch_size, 3, 320, 320]

        return img_paths, imgs, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps