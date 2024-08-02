import copy
import random
from collections import defaultdict

import numpy as np
import torch
from datasets.transforms import Transformations
from datasets.vehicle_id import VehicleID
from datasets.veri_776 import Veri776
from datasets.veri_wild import VeriWild
from misc.utils import read_image
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

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
        
        # Transformations for the dataset
        self.transforms = Transformations(dataset=self.dataset_name)
        
        # Create the train and validation datasets
        self.train_set = ImageDataset(self.dataset.train, transform=self.transforms.get_train_transform())
        self.validation_set = ImageDataset(self.dataset.query + self.dataset.gallery, transform=self.transforms.get_val_transform())

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

    def __getitem__(self, idx) -> tuple[str, torch.Tensor, int, int, int, int, int, str]:
        img_path, car_id, cam_id, model_id, color_id, type_id, timestamp = self.data[idx]

        # Read image in PIL format
        img = read_image(img_path)

        # Apply the transform if it exists
        if self.transform is not None:
            img = self.transform(img)

        return img_path, img, car_id, cam_id, model_id, color_id, type_id, timestamp

    def train_collate_fn(self, batch) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, str]:
        img_paths, imgs, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps = zip(*batch)

        # Transform Car IDs and Images to Tensors
        imgs = torch.stack(imgs, dim=0)                     # [batch_size, 3, 320, 320]
        car_ids = torch.tensor(car_ids, dtype=torch.int64)  # [batch_size]
        cam_ids = torch.tensor(cam_ids, dtype=torch.int64)

        return img_paths, imgs, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps
    
    def val_collate_fn(self, batch) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, str]:
        img_paths, imgs, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps = zip(*batch)

        # Transform Car IDs and Images to Tensors
        imgs = torch.stack(imgs, dim=0)                     # [batch_size, 3, 320, 320]
        car_ids = torch.tensor(car_ids, dtype=torch.int64)  # [batch_size]
        cam_ids = torch.tensor(cam_ids, dtype=torch.int64)

        return img_paths, imgs, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - batch_size (int): number of examples in a batch.
    - num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        
        # img_path, car_id, cam_id, model_id, color_id, type_id, timestamp
        for index, (_, pid, _, _, _, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
