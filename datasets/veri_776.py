import glob
import os
import re
import xml.etree.ElementTree as ET

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