from collections import defaultdict
import os
import pickle
import random
import shutil

class VehicleID():
    def __init__(self, data_path, dataset_size='small', use_rptm=False):
        # Generic variables
        self.dataset_name = 'vehicle_id'
        self.data_path = os.path.join(data_path, 'VehicleID')
        
        self.dataset_sizes = {'small': 800, 'medium': 1600, 'large': 2400} # Number of test images - 800 (Small), 1600 (Medium), 2400 (Large)
        self.test_size = self.dataset_sizes[dataset_size]

        # Vehicle Infos
        self.vehicle_infos = self.get_vehicle_infos(os.path.join(self.data_path, 'attribute'))
        
        # Read color IDs
        self.color_dict = {}
        with open(os.path.join(self.data_path, 'attribute', 'color_names.txt'), 'r') as file:
            for line in file:
                color, index = line.strip().split(' ', 1)
                self.color_dict[int(index)] = color.lower()

        # Directory
        self.img_dir = os.path.join(self.data_path, 'image')

        # Train, Test and Query
        self.train = self.get_list(os.path.join(self.data_path, 'train_test_split', 'train_list.txt'), relabel=True)

        # Get test size
        if self.test_size == 800:
            self.test_list = os.path.join(self.data_path, 'train_test_split', 'test_list_800.txt')
        elif self.test_size == 1600:
            self.test_list = os.path.join(self.data_path, 'train_test_split', 'test_list_1600.txt')
        elif self.test_size == 2400:
            self.test_list = os.path.join(self.data_path, 'train_test_split', 'test_list_2400.txt')
        else:
            raise ValueError(f"Unknown dataset size: {dataset_size}")

        # Test (query and gallery)
        self.gallery, self.query = self.get_query_gallery(self.test_list, relabel=True)

        # Retrieve the unique car IDs
        self.len = self.get_unique_car_ids()
        
        # Whether to init Dataset for RPTM Training or not
        self.use_rptm = use_rptm
        self.pkl_file = None
        self.max_negative_labels = 13164
        
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
                    s = name.split('.')[0]
                self.gms[s] = pickle.load(f)
                f.close

            # (img_path, folder, car_id, cam_id, model_id, color_id, type_id, timestamp)
            for _, folder, car_id, _, _, _, _, _ in self.train:
                self.pidx[folder] = car_id

            # Create the splits
            split_dir = os.path.join(self.data_path, 'splits')
            if not os.path.exists(split_dir):
                print("Splits not found. Creating splits for RPTM training...")
                src_root = os.path.join(self.data_path, 'image')
                for i in os.listdir(src_root):
                    folder_name = i.split('_', 2)[0]
                    if not os.path.exists(os.path.join(split_dir, folder_name)):
                        os.makedirs(os.path.join(split_dir, folder_name))
                    shutil.copyfile(os.path.join(src_root, i), os.path.join(split_dir, folder_name, i))

    def get_vehicle_infos(self, file_path):
        # Image to Vehicle ID mapping
        img2vid = {}
        with open(os.path.join(file_path, 'img2vid.txt'), 'r') as f:
            for line in f:
                img, vid = line.strip().split()
                if int(vid) == -1:
                    continue  # junk images are just ignored
                else:
                    assert 0 <= int(vid) <= 27957 # pid == 0 means background
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

    def get_list(self, file_path, relabel=True):
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
                img_path = (os.path.join(self.img_dir, line[0] + '.jpg'))
                car_id = vehicle_info['car_id']

                if car_id == -1:
                    continue  # junk images are just ignored
                assert 0 <= car_id <= 27957  # pid == 0 means background

                folder = str(car_id)
                if (len(folder) == 1):
                    folder = '0000' + folder
                elif (len(folder) == 2):
                    folder = '000' + folder
                elif (len(folder) == 3):
                    folder = '00' + folder
                elif (len(folder) == 4):
                    folder = '0' + folder
                else:
                    pass

                cam_id = -1  # Camera ID is not available
                model_id = vehicle_info['model_id'] # model ID is the model name
                color_id = vehicle_info['color_id'] # Color ID is the color name
                type_id = -1  # Type ID is not available
                timestamp = -1  # Timestamp is not available
                dataset.append((img_path, folder, car_id, cam_id, model_id, color_id, type_id, timestamp))
            except StopIteration:
                print(
                    f"No matching details found for Vehicle with ID: {vehicle_id}")
                
        # Before returning the dataset, relabel the IDs if required
        if relabel:
            return self.relabel_ids(dataset)
        else:
            return dataset

    def relabel_ids(self, dataset):
        all_pids = {}
        all_folders = {}
        relabeled_dataset = []
        for item in dataset:
            img_path, folder, car_id, camera_id, model_id, color_id, type_id, timestamp = item

            # Relabel the car IDs
            if car_id not in all_pids:
                all_pids[car_id] = len(all_pids)

            # Relabel also folders
            if folder not in all_folders:
                all_folders[folder] = len(all_folders) + 1 # Start from 1 (00001, ..)
                temp_folder = str(all_folders[folder])
                if (len(temp_folder) == 1):
                    temp_folder = '0000' + temp_folder
                elif (len(temp_folder) == 2):
                    temp_folder = '000' + temp_folder
                elif (len(temp_folder) == 3):
                    temp_folder = '00' + temp_folder
                elif (len(temp_folder) == 4):
                    temp_folder = '0' + temp_folder
                else:
                    pass
                all_folders[folder] = temp_folder
            
            new_id = all_pids[car_id]
            new_folder = all_folders[folder]
            
            relabeled_dataset.append((img_path, new_folder, new_id, camera_id, model_id, color_id, type_id, timestamp))
        return relabeled_dataset

    def get_query_gallery(self, file_path, relabel=True):
        test_pid_dict = defaultdict(list)

        with open(file_path) as f_test:
            test_data = f_test.readlines()
            for data in test_data:
                name, pid = data.strip().split(' ')
                test_pid_dict[pid].append([name, pid])
        test_pids = list(test_pid_dict.keys())
        num_test_pids = len(test_pids)

        assert num_test_pids == self.test_size, 'There should be {} vehicles for testing,' \
                                                ' but but got {}, please check the data'\
                                                .format(self.test_size, num_test_pids)

        query_data = []
        gallery_data = []

        # for each test id, random choose one image for gallery
        # and the other ones for query.
        for pid in test_pids:
            imginfo = test_pid_dict[pid]
            sample = random.choice(imginfo)
            imginfo.remove(sample)
            gallery_data.extend(imginfo)
            query_data.append(sample)

        query = self.parse_img_pids(query_data, relabel=relabel)
        gallery = self.parse_img_pids(gallery_data, relabel=relabel)
        return query, gallery

    def parse_img_pids(self, nl_pairs, relabel=False):
        # il_pair is the pairs of img name and label
        output = []
        for info in nl_pairs:
            name = info[0]
            car_id = info[1]
            img_path = os.path.join(self.img_dir, name + '.jpg')
            folder, cam_id, model_id, type_id, color_id, timestamp = -1, -1, -1, -1, -1, -1

            try:
                vehicle_info = self.vehicle_infos[name]
                car_id = vehicle_info['car_id']

                folder = str(car_id)
                if (len(folder) == 1):
                    folder = '0000' + folder
                elif (len(folder) == 2):
                    folder = '000' + folder
                elif (len(folder) == 3):
                    folder = '00' + folder
                elif (len(folder) == 4):
                    folder = '0' + folder
                else:
                    pass

                cam_id = -1  # Camera ID is not available
                model_id = vehicle_info['model_id'] # model ID is the model name
                color_id = vehicle_info['color_id'] # Color ID is the color name
                type_id = -1  # Type ID is not available
                timestamp = -1  # Timestamp is not available
            except StopIteration:
                print(
                    f"No matching details found for Vehicle with ID: {car_id}")
            output.append((img_path, folder, car_id, cam_id, model_id, color_id, type_id, timestamp))

        # Before returning the dataset, relabel the IDs if required
        if relabel:
            return self.relabel_ids(output)
        else:
            return output

    def get_unique_car_ids(self):
        # Combine all car IDs from train
        all_car_ids = {car_id for _, _, car_id, _, _, _, _, _ in self.train}
        # Get the unique car IDs
        return len(all_car_ids) + 1 # Add 1 to account for the 0-indexing
    
    def get_color_index(self, color_pred: str):
        for index, color in self.color_dict.items():
            if color == color_pred:
                return index
            
        # If the code reaches this point, it means that the color is not found
        # Hence, we would like to add it to the list of colors
        color_index = len(self.color_dict) + 1
        self.color_dict[color_index] = color_pred
        return color_index
