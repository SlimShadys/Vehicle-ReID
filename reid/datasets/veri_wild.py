import os
import pickle
import shutil

class VeriWild():
    def __init__(self, data_path, dataset_size='small', use_rptm=False):
        # Generic variables
        self.dataset_name = 'veri_wild'
        self.data_path = os.path.join(data_path, 'VeRi-Wild')
        self.dataset_sizes = {'small': 3000, 'medium': 5000, 'large': 10000} # Number of test images - 3000 (Small), 5000 (Medium), 10000 (Large)
        self.test_size = self.dataset_sizes[dataset_size]

        # Vehicle Infos
        self.vehicle_infos = self.get_vehicle_infos(os.path.join(self.data_path, 'train_test_split', 'vehicle_info.txt'))

        # Directory
        self.img_dir = os.path.join(self.data_path, 'images')

        # Train, Test and Query
        self.train = self.get_list(os.path.join(self.data_path, 'train_test_split', 'train_list_start0.txt'))
        self.gallery = self.get_list(os.path.join(self.data_path, 'train_test_split', f'test_{self.test_size}_id.txt'))
        self.query = self.get_list(os.path.join(self.data_path, 'train_test_split', f'test_{self.test_size}_id_query.txt'))

        # Relabel the train set only while keeping the original IDs for the query and gallery sets
        self.train = self.relabel_ids(self.train)

        # Number of unique car IDs
        self.len = self.get_unique_car_ids()

        # Whether to init Dataset for RPTM Training or not
        self.use_rptm = use_rptm
        self.pkl_file = None
        self.max_negative_labels = 40671

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

            # Create the splits directory if it does not exist
            split_dir = os.path.join(self.data_path, 'splits')
            if not os.path.exists(split_dir):
                print("Splits not found. Creating splits for RPTM training...")
                for path, folder, car_id, _, _, _, _, _ in self.train:
                    split_dir = os.path.join(self.data_path, 'splits', folder)
                    if not os.path.exists(split_dir):
                        os.makedirs(split_dir)
                    shutil.copyfile(path, os.path.join(split_dir, path.split(os.path.sep)[-1]))

    def relabel_ids(self, dataset):
        all_pids = {}
        relabeled_dataset = []
        for item in dataset:
            img_path, folder, vehicle_id, camera_id, model_id, color_id, type_id, timestamp = item

            if vehicle_id not in all_pids:
                all_pids[vehicle_id] = len(all_pids)

            new_id = all_pids[vehicle_id] + 1
            
            # Relabel the folder name with the new_id
            folder = str(new_id)
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
            
            relabeled_dataset.append((img_path, folder, new_id, camera_id, model_id, color_id, type_id, timestamp))
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
        self.color_dict = {}
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
                if color_name not in self.color_dict.values():
                    idx = len(self.color_dict)
                    self.color_dict[idx] = color_name
                if type_name not in type_dict:
                    type_dict[type_name] = len(type_dict)

                item['model_ID'] = model_dict[model_name]
                for index, color in self.color_dict.items():
                    if color == color_name:
                        item['color_ID'] = index
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
            vehicle_id = line[0].split('/')[0]
            vehicle_img = line[0].split('/')[1].split('.')[0]
            vehicle_info = self.vehicle_infos[int(vehicle_id)]

            # Extract the details
            try:
                result = next(row for row in vehicle_info if row['vehicle_img'] == int(vehicle_img))
            except StopIteration:
                print(f"No matching row found for vehicle_img {int(vehicle_img)}")

            # Create the folder name, starting from the car ID
            folder = str(result['vehicle_ID'])
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

            # (img_path, folder, car_id, cam_id, model_id, color_id, type_id, timestamp)
            dataset.append((os.path.join(self.img_dir, folder, vehicle_img + '.jpg'), # img_path
                            folder, # Folder
                            result['vehicle_ID'],
                            result['camera_ID'],
                            # result['model_name'] -> If we want the string name of the model
                            result['model_ID'],
                            # result['color_name'] -> If we want the string name of the color
                            result['color_ID'],
                            # result['type_name'] -> If we want the string name of the type
                            result['type_ID'],
                            result['timestamp']))
        return dataset

    def get_unique_car_ids(self):
        # Combine all car IDs from train, query, and gallery sets
        all_car_ids = {car_id for _, _, car_id, _, _, _, _, _ in self.train}
        # Get the unique car IDs
        return len(all_car_ids)
    
    def get_color_index(self, color_pred: str):
        for index, color in self.color_dict.items():
            if color == color_pred:
                return index
            
        # If the code reaches this point, it means that the color is not found
        # Hence, we would like to add it to the list of colors
        color_index = len(self.color_dict) + 1
        self.color_dict[color_index] = color_pred
        return color_index