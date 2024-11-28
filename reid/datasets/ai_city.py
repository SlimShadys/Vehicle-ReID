import glob
import os
import pickle
import re
import shutil
import xml.etree.ElementTree as ET

class AICity():
    def __init__(self, data_path, use_rptm=False, use_sim=False):
        # Generic variables
        self.dataset_name = 'ai_city'
        self.use_sim = use_sim
        self.data_path = os.path.join(data_path, 'AIC21_Track2_ReID')
        self.train_labels = self.get_labels(file_path=os.path.join(self.data_path, 'train_label.xml'), sim=False)
        self.test_labels = self.get_labels(file_path=os.path.join(self.data_path, 'test_label.xml'), sim=False)
        self.query_labels = self.get_labels(file_path=os.path.join(self.data_path, 'query_label.xml'), sim=False)

        # Directories
        self.train_dir = os.path.join(self.data_path, 'image_train')
        self.query_dir = os.path.join(self.data_path, 'image_query')
        self.gallery_dir = os.path.join(self.data_path, 'image_test')

        # Train
        self.train = self.get_list(self.train_labels, self.train_dir, sim=False)

        # Test (query and gallery)
        self.gallery = self.get_list(self.test_labels, self.gallery_dir, sim=False)

        # No details for the query set
        self.query = self.get_list(self.query_labels, self.query_dir, sim=False)

        if self.use_sim:
            self.sim_data_path = os.path.join(data_path, 'AIC21_Track2_ReID_Simulation')
            self.sim_train_labels = self.get_labels(file_path=os.path.join(self.sim_data_path, 'train_label.xml'), sim=True)
            self.sim_train_dir = os.path.join(self.sim_data_path, 'sys_image_train')
            
            self.train_sim = self.get_list(self.sim_train_labels, self.sim_train_dir, sim=True)
            self.train += self.train_sim 

        # Number of unique car IDs
        self.len = self.get_unique_car_ids()

        # Whether to init Dataset for RPTM Training or not
        self.use_rptm = use_rptm
        self.pkl_file = None
        self.max_negative_labels = self.len

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

    def get_labels(self, file_path, sim):
        # Open the file with the correct encoding
        with open(file_path, 'r', encoding='gb2312') as file:
            xml_content = file.read()

        # Parse the XML string
        root = ET.fromstring(xml_content)

        # Dictionary to store all items
        labels_dict = {}

        if sim:
            len_train = max([v['vehicle_ID'] for k, v in self.train_labels.items()])

        # Iterate through each Item element
        for item in root.findall('.//Item'):
            # Extract the image name
            image_name = item.get('imageName')

            # Extract numeric part of cameraID
            camera_id_str = item.get('cameraID')
            camera_id_num = int(re.search(r'\d+', camera_id_str).group()) - 1 # Subtract 1 to start from 0

            # Extract the vehicle ID
            vehicle_id_str = item.get('vehicleID')
            try:
                car_id_num = int(re.search(r'\d+', vehicle_id_str).group()) - 1 # Subtract 1 to start from 0
                if sim:
                    car_id_num += len_train + 1
            except:
                car_id_num = -2 # Test and Query images do not have vehicle ID

            # Various checks to ensure the data is correct
            if car_id_num == -1:
                continue  # Junk images are just ignored

            item_info = {
                'vehicle_ID': car_id_num,
                'camera_ID': camera_id_num,
                'model_ID': -1, # Model ID is not available in the XML file
                'type_ID': -1,
                'color_ID': -1,
                'timestamp': 'None'
            }
            labels_dict[image_name] = item_info

        return labels_dict

    def get_list(self, labels, dir_path, sim=False):
        dataset = []
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))

        for img_path in img_paths:
            # Details
            car_detail = labels[os.path.basename(img_path)]
            # Extract folder value (Needed for RPTM)
            folder = (img_path.split(os.path.sep, 4)[-1]).split('.')[0]
            car_id = car_detail['vehicle_ID']
            cam_id = car_detail['camera_ID']

            # Various checks to ensure the data is correct
            if car_id == -1:
                continue  # Junk images are just ignored
            
            model_id = car_detail['model_ID']
            type_id = car_detail['type_ID']
            color_id = car_detail['color_ID']
            timestamp = car_detail['timestamp']
            details = (img_path, folder, car_id, cam_id, model_id, color_id, type_id, timestamp)

            dataset.append(details)

        # Relabel the train set only
        if 'train' in dir_path:
            dataset = self.relabel_ids(dataset, sim)

        return dataset

    def relabel_ids(self, dataset, sim=False):
        all_pids = {}
        relabeled_dataset = []

        if sim:
            len_train = max([v['vehicle_ID'] for k, v in self.train_labels.items()]) + 1

        for item in dataset:
            img_path, folder, vehicle_id, camera_id, model_id, color_id, type_id, timestamp = item

            if vehicle_id not in all_pids:

                if sim:
                    all_pids[vehicle_id] = len(all_pids) + len_train
                else:
                    all_pids[vehicle_id] = len(all_pids)

            new_id = all_pids[vehicle_id]
            relabeled_dataset.append((img_path, folder, new_id, camera_id, model_id, color_id, type_id, timestamp))
        return relabeled_dataset

    def get_unique_car_ids(self):
        # Combine all car IDs from train, query, and gallery sets
        all_car_ids = {car_id for _, _, car_id, _, _, _, _, _ in self.train}
        # Get the unique car IDs
        return len(all_car_ids) + 1 # Add 1 to account for the 0-indexing
    
class AICitySim():
    def __init__(self, data_path, use_rptm=False):
        # Generic variables
        self.dataset_name = 'ai_city_sim'
        self.data_path = os.path.join(data_path, 'AIC21_Track2_ReID_Simulation')
        self.train_labels = self.get_labels(file_path=os.path.join(self.data_path, 'train_label.xml'))

        # Directories
        self.train_dir = os.path.join(self.data_path, 'sys_image_train')

        # Train
        self.train = self.get_list(self.train_labels, self.train_dir)

        # No Gallery and Query for the Simulation dataset
        self.query = []
        self.gallery = []

        # Number of unique car IDs
        self.len = self.get_unique_car_ids()

        # Whether to init Dataset for RPTM Training or not
        self.use_rptm = use_rptm
        self.pkl_file = None
        self.max_negative_labels = self.len

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
            # Extract the image name
            image_name = item.get('imageName')

            # Extract numeric part of cameraID
            camera_id_str = item.get('cameraID')
            camera_id_num = int(re.search(r'\d+', camera_id_str).group()) - 1 # Subtract 1 to start from 0

            # Extract the vehicle ID
            vehicle_id_str = item.get('vehicleID')
            car_id_num = int(re.search(r'\d+', vehicle_id_str).group()) - 1 # Subtract 1 to start from 0

            # Various checks to ensure the data is correct
            if car_id_num == -1:
                continue  # Junk images are just ignored

            item_info = {
                'vehicle_ID': car_id_num,
                'camera_ID': camera_id_num,
                'model_ID': -1, # Model ID is not available in the XML file
                'type_ID': -1,
                'color_ID': -1,
                'timestamp': 'None'
            }
            labels_dict[image_name] = item_info

        return labels_dict

    def get_list(self, labels, dir_path):
        dataset = []
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))

        for img_path in img_paths:
            # Details
            car_detail = labels[os.path.basename(img_path)]
            # Extract folder value (Needed for RPTM)
            folder = (img_path.split(os.path.sep, 4)[-1]).split('_')[1].split('c')[-1]
            car_id = car_detail['vehicle_ID']
            cam_id = car_detail['camera_ID']

            # Various checks to ensure the data is correct
            if car_id == -1:
                continue  # Junk images are just ignored
            
            model_id = car_detail['model_ID']
            type_id = car_detail['type_ID']
            color_id = car_detail['color_ID']
            timestamp = car_detail['timestamp']
            details = (img_path, folder, car_id, cam_id, model_id, color_id, type_id, timestamp)

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

    def get_unique_car_ids(self):
        # Combine all car IDs from train, query, and gallery sets
        all_car_ids = {car_id for _, _, car_id, _, _, _, _, _ in self.train}
        # Get the unique car IDs
        return len(all_car_ids) + 1 # Add 1 to account for the 0-indexing