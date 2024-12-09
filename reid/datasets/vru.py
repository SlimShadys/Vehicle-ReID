import json
import os

import numpy as np

class VRU(object):
    def __init__(self, data_path, dataset_size='small', split_id=0, use_rptm=False):
        super(VRU, self).__init__()
        # Generic variables
        self.dataset_name = 'vru'
        self.data_path = os.path.join(data_path, 'VRU')
        run_preprocess = True

        # Number of test images - 1200, 2400, 8000
        self.dataset_sizes = {'small': 1200, 'medium': 2400, 'large': 8000}

        # Get test size
        self.test_size = self.dataset_sizes[dataset_size]

        # Test list and split labeled JSON path
        if self.test_size == 1200:
            self.test_list = 'test_list_1200.txt'
            self.split_labeled_json_path = os.path.join(self.data_path, 'test_1200.json')
        elif self.test_size == 2400:
            self.test_list = 'test_list_2400.txt'
            self.split_labeled_json_path = os.path.join(self.data_path, 'test_2400.json')
        elif self.test_size == 8000:
            self.test_list = 'test_list_8000.txt'
            self.split_labeled_json_path = os.path.join(self.data_path, 'test_8000.json')
        else:
            raise ValueError(f"Unknown dataset size: {dataset_size}")

        # Directories
        self.label_dir = os.path.join(self.data_path, 'train_test_split')
        self.imgs_dir = os.path.join(self.data_path, 'Pic')

        # Other splits (from original code)
        # self.split_labeled_json_path = os.path.join(self.data_path, 'train_validation.json')
        # self.split_labeled_json_path = os.path.join(self.data_path, 'test_1200_big_gallery.json')
        # self.split_labeled_json_path = os.path.join(self.data_path, 'test_2400.json')
        # self.split_labeled_json_path = os.path.join(self.data_path, 'test_2400_big_gallery.json')
        # self.split_labeled_json_path = os.path.join(self.data_path, 'test_8000.json')
        # self.split_labeled_json_path = os.path.join(self.data_path, 'test_8000_big_gallery.json')
        # self.split_labeled_json_path = os.path.join(self.data_path, 'validation.json')
        # self.split_labeled_json_path = os.path.join(self.data_path, 'validation_big_gallery.json')

        # Check before running pre-processing
        self._check_before_run()

        # RPTM Support
        self.use_rptm = use_rptm
        self.pkl_file = None
        self.max_negative_labels = 7086

        # Proper pre-processing
        if run_preprocess:
            self._preprocess()

        # Load the data
        with open(self.split_labeled_json_path, 'r') as f:
            splits = json.load(f)
        assert split_id < len(splits), "Condition split_id ({}) < len(splits) ({}) is false".format(
            split_id, len(splits))
        split = splits[split_id]
        print("Split index = {}".format(split_id))

        self.train = split['train']  # list
        self.query = split['query']  # list
        self.gallery = split['gallery']  # list

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.data_path):
            raise RuntimeError("'{}' is not available".format(self.data_path))
        if not os.path.exists(self.label_dir):
            raise RuntimeError("'{}' is not available".format(self.label_dir))
        if not os.path.exists(self.imgs_dir):
            raise RuntimeError("'{}' is not available".format(self.imgs_dir))
        # if not os.path.exists(self.split_labeled_json_path):
        #     raise RuntimeError("'{}' is not available".format(self.split_labeled_json_path))

    def extract_split(self, label_dir, label_file_name, pic_dir, split_name):
        split_path = os.path.join(label_dir, label_file_name)
        with open(split_path, 'r') as split_file:
            lines = split_file.readlines()

        train, query, gallery = [], [], []
        car_dic = {}

        # Processing training set
        if split_name == "train":
            car_count = 0
            for line in lines:
                line_list = line.split()
                car_label = line_list[1]

                if car_label not in car_dic:
                    car_dic[car_label] = car_count
                    car_count += 1
                car_id = car_dic[car_label]

                img_info = [os.path.join(pic_dir, line_list[0] + ".jpg"),  # img_path
                                -1,  # folder
                            car_id,  # car ID
                                -1,  # cam ID
                                -1,  # model ID
                                -1,  # color ID
                                -1,  # type ID
                            "None",  # timestamp
                ]
                train.append(img_info)
            split_file.close()
            return train, len(car_dic), len(train)

        # Processing query and gallery sets
        else:
            for line in lines:
                line_list = line.split()
                if line_list[1] in car_dic:
                    car_dic[line_list[1]].append(int(line_list[0]))
                else:
                    car_dic[line_list[1]] = [int(line_list[0])]
            split_file.close()

            num_cids = 0
            car_id = 0
            for car_label, img_index in car_dic.items():
                if len(img_index) > 1:
                    num_cids += 1
                    # choose a random image to set the gallery
                    gallery_index = np.random.randint(0, len(img_index))

                    # We want to return a list like this:
                    # [img_path, folder, car_id, cam_id, model_id, color_id, type_id, timestamp]
                    gallery_img = [os.path.join(pic_dir, str(img_index[gallery_index]) + ".jpg"), # img_path
                                       -1,  # folder
                                   car_id,  # car ID
                                       -1,  # cam ID
                                       -1,  # model ID
                                       -1,  # color ID
                                       -1,  # type ID
                                   "None",  # timestamp
                                   ]
                    
                    # gallery.append(gallery_img)  # 图库集数量少
                    query.append(gallery_img)  # 查询集数量少

                    # Put the rest images into the query
                    img_index.pop(gallery_index)
                    
                    for i in range(len(img_index)):
                        query_img = [os.path.join(pic_dir, str(img_index[i]) + ".jpg"), # img_path
                                        -1,  # folder
                                    car_id,  # car ID
                                        -1,  # cam ID
                                        -1,  # model ID
                                        -1,  # color ID
                                        -1,  # type ID
                                    "None",  # timestamp
                                    ]
                        # query.append(query_img)
                        gallery.append(query_img)
                car_id += 1
            return query, gallery, num_cids, num_cids, len(query), len(gallery)

    def _preprocess(self):
        print("Extracting the splits ...")
        
        # Train
        train, num_train_cids, num_train_imgs = \
            self.extract_split(self.label_dir, "train_list.txt", self.imgs_dir, "train")

        # Query and Gallery
        query, gallery, num_query_cids, num_gallery_cids, num_query_imgs, num_gallery_imgs = \
            self.extract_split(self.label_dir, self.test_list, self.imgs_dir, "test")

        splits = [{
            'train': train,
            'query': query,
            'gallery': gallery,
        }]

        # Save the data to a JSON file
        with open(self.split_labeled_json_path, 'w') as f:
            json.dump(splits, f, indent=4, separators=(',', ': '))

    def get_unique_car_ids(self):
        # Combine all car IDs from train, query, and gallery sets
        all_car_ids = {car_id for _, _, car_id, _, _, _, _, _ in self.train}
        # Get the unique car IDs
        return len(all_car_ids) # Add 1 to account for the 0-indexing