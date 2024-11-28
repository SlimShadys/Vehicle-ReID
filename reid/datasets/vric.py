import glob
import os
import pickle
import shutil

class VRIC():
    def __init__(self, data_path, use_rptm=False):
        # Generic variables
        self.dataset_name = 'vric'
        self.data_path = os.path.join(data_path, 'VRIC')
        self.train_labels = self.get_labels(file_path=os.path.join(self.data_path, 'vric_train.txt'))
        self.test_labels = self.get_labels(file_path=os.path.join(self.data_path, 'vric_gallery.txt'))
        self.query_labels = self.get_labels(file_path=os.path.join(self.data_path, 'vric_probe.txt'))

        # Directories
        self.train_dir = os.path.join(self.data_path, 'train_images')
        self.query_dir = os.path.join(self.data_path, 'probe_images')
        self.gallery_dir = os.path.join(self.data_path, 'gallery_images')

        # Train
        self.train = self.get_list(self.train_labels, self.train_dir)

        # Test (query and gallery)
        self.gallery = self.get_list(self.test_labels, self.gallery_dir)

        # No details for the query set
        self.query = self.get_list(self.query_labels, self.query_dir)

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
        # Dictionary to store all items
        labels_dict = {}

        # Read the .txt file
        with open(file_path, 'r') as file:
            lines = file.readlines()

            for line in lines:
                # [Image_name] [ID label] [Cam Label]
                line = line.strip().split()

                # Get the image name
                image_name = str(line[0])

                # Get the vehicle ID
                vehicle_id = int(line[1]) - 1 # Subtract 1 to start from 0

                # Get the camera ID
                camera_id = int(line[2]) - 1 # Subtract 1 to start from 0

                # Various checks to ensure the data is correct
                if vehicle_id == -1:
                    continue  # Junk images are just ignored

                item_info = {
                    'vehicle_ID': vehicle_id,
                    'camera_ID': camera_id,
                    'model_ID': -1, # Model ID is not available in the XML file
                    'type_ID': -1,
                    'color_ID': -1, # Subtract 1 to start from 0
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
            folder = (img_path.split(os.path.sep, 4)[-1]).split('_')[2]
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