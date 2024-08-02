import os

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