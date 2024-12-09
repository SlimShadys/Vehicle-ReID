import glob
import os
import pickle
import re
import shutil
import xml.etree.ElementTree as ET

class Veri776():
    def __init__(self, data_path, use_rptm=False):
        # Generic variables
        self.dataset_name = 'veri_776'
        self.data_path = os.path.join(data_path, 'VeRi-776')
        self.train_labels = self.get_labels(os.path.join(self.data_path, 'train_label.xml'))
        self.test_labels = self.get_labels(os.path.join(self.data_path, 'test_label.xml'))

        # Read color IDs
        self.color_dict = {}
        with open(os.path.join(self.data_path, 'list_color.txt'), 'r') as file:
            for line in file:
                index, color = line.strip().split(' ', 1)
                index = int(index) - 1 # Subtract 1 to start from 0
                self.color_dict[index] = color.lower()

        # Directories
        self.train_dir = os.path.join(self.data_path, 'image_train')
        self.query_dir = os.path.join(self.data_path, 'image_query')
        self.gallery_dir = os.path.join(self.data_path, 'image_test')

        # Train
        self.train = self.get_list(self.train_labels, self.train_dir)

        # Test (query and gallery)
        self.gallery = self.get_list(self.test_labels, self.gallery_dir)

        # No details for the query set
        self.query = self.get_list(self.test_labels, self.query_dir)

        # Junk index and GT index
        self.indices = self.get_indices('name_query.txt', 'jk_index.txt', 'gt_index.txt')

        # Number of unique car IDs
        self.len = self.get_unique_car_ids()

        # Whether to init Dataset for RPTM Training or not
        self.use_rptm = use_rptm
        self.pkl_file = None
        self.max_negative_labels = 770
        
        if (self.use_rptm):
            self.gms = {}
            self.pidx = {}

            self.pkl_file = f'index_vp_{self.dataset_name}.pkl'
            gms_path = os.path.join('gms', self.dataset_name)
            entries = sorted(os.listdir(gms_path))

            # Loop through the files and load the GMS features
            for name in entries:
                f = open((os.path.join(gms_path, name)), 'rb')
                if name == 'featureMatrix.pkl':
                    s = name[0:13]
                else:
                    s = name[0:3]
                self.gms[s] = pickle.load(f)
                f.close

            # (img_path, folder, car_id, cam_id, model_id, color_id, type_id, timestamp)
            for _, folder, car_id, _, _, _, _, _ in self.train:
                self.pidx[folder] = car_id

            # Create the splits directory if it does not exist
            split_dir = os.path.join(self.data_path, 'splits')
            if not os.path.exists(split_dir):
                print("Splits not found. Creating splits for RPTM training...")
                src_root = os.path.join(self.data_path, 'image_train')
                for i in os.listdir(src_root):
                    folder_name = i.split('_', 2)[0][1:]
                    if not os.path.exists(os.path.join(split_dir, folder_name)):
                        os.makedirs(os.path.join(split_dir, folder_name))
                    shutil.copyfile(os.path.join(src_root, i), os.path.join(split_dir, folder_name, i))

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
            # Subtract 1 to start from 0
            camera_id_num = int(re.search(r'\d+', camera_id_str).group()) - 1
            # Subtract 1 to start from 0
            car_id_num = int(re.search(r'\d+', vehicle_id_str).group()) - 1

            # Various checks to ensure the data is correct
            if car_id_num == -1:
                continue  # Junk images are just ignored

            item_info = {
                'vehicle_ID': car_id_num,
                'camera_ID': camera_id_num,
                'model_ID': -1, # Model ID is not available in the XML file
                'type_ID': int(item.get('typeID')),
                'color_ID': int(item.get('colorID')) - 1, # Subtract 1 to start from 0
                'timestamp': 'None'
            }
            labels_dict[item.get('imageName')] = item_info

        return labels_dict

    def get_list(self, labels, dir_path):
        dataset = []
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))

        for img_path in img_paths:
            try:
                # Details
                car_detail = labels[os.path.basename(img_path)]
                # Extract folder value (Needed for RPTM)
                folder = (img_path.split(os.path.sep, 4)[-1]).split('_', 1)[0][1:]
                car_id = car_detail['vehicle_ID']
                cam_id = car_detail['camera_ID']

                # Various checks to ensure the data is correct
                if car_id == -1:
                    continue  # Junk images are just ignored
                assert 0 <= car_id <= 1501 # pid == 0 means background
                assert 0 <= cam_id <= 19   # Check if the camera ID is within the range
                
                model_id = car_detail['model_ID']
                type_id = car_detail['type_ID']
                color_id = car_detail['color_ID']
                timestamp = car_detail['timestamp']
                details = (img_path, folder, car_id, cam_id, model_id, color_id, type_id, timestamp)
            except:
                # There are 32 images that are not in the XML file,
                # So we need to extract the infos manually from the image name
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
            img_path, folder, vehicle_id, camera_id, model_id, color_id, type_id, timestamp = item

            if vehicle_id not in all_pids:
                all_pids[vehicle_id] = len(all_pids)

            new_id = all_pids[vehicle_id]
            relabeled_dataset.append((img_path, folder, new_id, camera_id, model_id, color_id, type_id, timestamp))
        return relabeled_dataset

    def retrieve_details(self, img_path: str):
        # We need to extract the infos manually from the image name
        # such as: '0002_c004_00084250_0.jpg'
        #
        # This should happen only when we are dealing with the query set
        # or, in general, if we want to recover the details from the image name
        img_full_string = os.path.basename(img_path).split('_')

        # Extract folder value (Needed for RPTM)
        folder = (img_path.split(os.path.sep, 4)[-1]).split('_', 1)[0][1:]

        # Extract the vehicle and camera IDs
        # Subtract 1 to start from 0
        car_id_num = int(re.search(r'\d+', img_full_string[0]).group()) - 1
        # Subtract 1 to start from 0
        camera_id_num = int(re.search(r'\d+', img_full_string[1]).group()) - 1

        # Details | Color ID and Type ID are not available in the image name, so we set them to -1
        car_id = car_id_num
        cam_id = camera_id_num
        model_id = -1
        type_id = -1
        color_id = -1
        timestamp = 'None'

        return (img_path, folder, car_id, cam_id, model_id, color_id, type_id, timestamp)

    def get_unique_car_ids(self):
        # Combine all car IDs from train, query, and gallery sets
        all_car_ids = {car_id for _, _, car_id, _, _, _, _, _ in self.train}
        # Get the unique car IDs
        return len(all_car_ids) # Add 1 to account for the 0-indexing
    
    def get_color_index(self, color_pred: str):
        for index, color in self.color_dict.items():
            if color == color_pred:
                return index
            
        # If the code reaches this point, it means that the color is not found
        # Hence, we would like to add it to the list of colors
        color_index = len(self.color_dict) + 1
        self.color_dict[color_index] = color_pred
        return color_index
