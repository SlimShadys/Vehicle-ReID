import copy
import os
import pickle
import random
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
import torch
from datasets.transforms import Transformations
from datasets.vehicle_id import VehicleID
from datasets.veri_776 import Veri776
from datasets.veri_wild import VeriWild
from datasets.vru import VRU
from misc.utils import read_image, get_imagedata_info
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

class DatasetBuilder():
    def __init__(self, data_path, dataset_name, dataset_size='small', use_rptm=False, augmentation_configs=None):
        """
        Args:
            data_path (string): Path to the data file.
            dataset_name (string): Name of the dataset.
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.dataset_size = dataset_size
        self.use_rptm = use_rptm # Whether to use RPTM Training

        # For veri_776 dataset, there is not a specific dataset type
        # For veri_wild, vehicle_id, and vru there are three types: small, medium, large
        if self.dataset_name == 'veri_776':
            self.dataset = Veri776(self.data_path, self.use_rptm)
        elif self.dataset_name == 'veri_wild':
            self.dataset = VeriWild(self.data_path, self.dataset_size, self.use_rptm)
        elif self.dataset_name == 'vehicle_id':
            self.dataset = VehicleID(self.data_path, self.dataset_size, self.use_rptm)
        elif self.dataset_name == 'vru':
            self.dataset = VRU(self.data_path, self.dataset_size, split_id=0, use_rptm=self.use_rptm)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Transformations for the dataset
        self.transforms = Transformations(
            dataset=self.dataset_name, configs=augmentation_configs)

        # Create the train and validation datasets
        self.train_set = ImageDataset(overall_dataset=self.dataset,
                                      data=self.dataset.train,
                                      pkl_file=self.dataset.pkl_file,
                                      transform=self.transforms.get_train_transform(),
                                      train=True)
        self.validation_set = ImageDataset(overall_dataset=self.dataset,
                                           data=self.dataset.query + self.dataset.gallery,
                                           pkl_file=None,
                                           transform=self.transforms.get_val_transform(),
                                           train=False)
        verbose = True
        if verbose:
            num_train_pids, num_train_imgs, num_train_cams = get_imagedata_info(self.dataset.train)
            num_query_pids, num_query_imgs, num_query_cams = get_imagedata_info(self.dataset.query)
            num_gallery_pids, num_gallery_imgs, num_gallery_cams = get_imagedata_info(self.dataset.gallery)

            print('Image Dataset statistics:')
            print('/------------------------------------------\\')
            print('|  Subset  |  # IDs | # Images | # Cameras |')
            print('|------------------------------------------|')
            print('|  Train   |  {:5d} | {:8d} | {:9d} |'.format(num_train_pids, num_train_imgs, num_train_cams))
            print('|  Query   |  {:5d} | {:8d} | {:9d} |'.format(num_query_pids, num_query_imgs, num_query_cams))
            print('|  Gallery |  {:5d} | {:8d} | {:9d} |'.format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
            print('\\------------------------------------------/')
        
class ImageDataset(Dataset):
    def __init__(self, overall_dataset: Union[VehicleID, Veri776, VeriWild], data, pkl_file, transform=None, train=True):
        """
        Args:
            overall_dataset (object): The overall dataset object.
            data (list): List of tuples containing the image path, folder, car_id, cam_id, model_id, color_id, type_id, timestamp.
            pkl_file (string): The pickle file containing the indices for the dataset. (ONLY for RPTM Training)
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): Whether the dataset is for training or validation.
        """
        self.train = train
        self.overall_dataset = overall_dataset
        self.dataset_name = overall_dataset.dataset_name
        self.pkl_file = pkl_file

        # Load the index from the pickle file (Needed for RPTM Training)
        if (self.overall_dataset.use_rptm) and (self.pkl_file is not None) and (self.train):
            with open(os.path.join('gms', self.dataset_name, self.pkl_file), 'rb') as handle:
                self.index = pickle.load(handle)

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[str, torch.Tensor, int, int, int, int, int, str]:
        img_path, folder, car_id, cam_id, model_id, color_id, type_id, timestamp = self.data[idx]

        # Read image in PIL format
        img = read_image(img_path)

        # Apply the transform if it exists
        if self.transform is not None:
            img = self.transform(img)

        # Custom indices for the dataset when using RPTM Training
        if (self.overall_dataset.use_rptm) and (self.train):
            if self.dataset_name == 'veri_776':
                index = 0 if self.data[idx][0] not in self.index else self.index[self.data[idx][0]][1]
            elif self.dataset_name == 'vehicle_id':
                index = self.index[self.data[idx][0]][1]
            elif self.dataset_name == 'veri_wild':
                index = 0 if self.data[idx][0].split(os.sep)[-1] not in self.index else self.index[self.data[idx][0].split(os.sep)[-1]][1]
            elif self.dataset_name == 'vru':
                raise NotImplementedError("RPTM Training is not fully implemented yet for VRU dataset.")
            else:
                index = 0
        else:
            index = 0

        return img_path, img, folder, index, car_id, cam_id, model_id, color_id, type_id, timestamp

    def train_collate_fn(self, batch) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, str]:
        img_paths, imgs, folders, indices, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps = zip(*batch)

        # Transform Car IDs and Images to Tensors
        # [batch_size, 3, height, width]
        imgs = torch.stack(imgs, dim=0)
        car_ids = torch.tensor(car_ids, dtype=torch.int64)  # [batch_size]
        cam_ids = torch.tensor(cam_ids, dtype=torch.int64)

        return img_paths, imgs, folders, indices, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps

    def val_collate_fn(self, batch) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, str]:
        img_paths, imgs, folders, indices, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps = zip(
            *batch)

        # Transform Car IDs and Images to Tensors
        imgs = torch.stack(imgs, dim=0)                     # [batch_size, 3, height, width]
        car_ids = torch.tensor(car_ids, dtype=torch.int64)  # [batch_size]
        cam_ids = torch.tensor(cam_ids, dtype=torch.int64)

        return img_paths, imgs, folders, indices, car_ids, cam_ids, model_ids, color_ids, type_ids, timestamps

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
        for index, (_, _, pid, _, _, _, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)  # Define num_identities here

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

class BalancedIdentitySampler(Sampler):
    def __init__(self, data_source: List, batch_size: int, num_instances: int, seed: Optional[int] = None):

        self.data_source = data_source
        self.num_instances = num_instances
        self.batch_size = batch_size

        self.num_pids_per_batch = batch_size // self.num_instances
        
        # Calculate the number of batches per epoch
        self.num_batches = len(self.data_source) // self.batch_size

        self.index_pid = dict()
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            camid = info[2]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = sorted(list(self.pid_index.keys()))
        self.num_identities = len(self.pids)

        if seed is None:
            seed = np.random.randint(2 ** 31)
        self._seed = int(seed)

        # Estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.pid_index[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def no_index(self, a, b):
        assert isinstance(a, list)
        return [i for i, j in enumerate(a) if j != b]

    def reorder_index(self, batch_indices, world_size):
        r"""Reorder indices of samples to align with DataParallel training.
        In this order, each process will contain all images for one ID, triplet loss
        can be computed within each process, and BatchNorm will get a stable result.
        Args:
            batch_indices: A batched indices generated by sampler
            world_size: number of process
        Returns:

        """
        mini_batchsize = len(batch_indices) // world_size
        reorder_indices = []
        for i in range(0, mini_batchsize):
            for j in range(0, world_size):
                reorder_indices.append(batch_indices[i + j * mini_batchsize])
        return reorder_indices

    def __len__(self):
        return self.length

    def __iter__(self):
        yield from self._finite_indices()

    def _finite_indices(self):
        np.random.seed(self._seed)
        batch_count = 0

        for _ in range(self.num_batches):
            batch_indices = []
            identities = np.random.permutation(self.num_identities)[:self.num_pids_per_batch]

            for kid in identities:
                i = np.random.choice(self.pid_index[self.pids[kid]])
                _, i_pid, i_cam, _, _, _, _ = self.data_source[i]
                batch_indices.append(i)
                pid_i = self.index_pid[i]
                cams = self.pid_cam[pid_i]
                index = self.pid_index[pid_i]
                select_cams = self.no_index(cams, i_cam)

                if select_cams:
                    if len(select_cams) >= self.num_instances:
                        cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
                    else:
                        cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)
                    for kk in cam_indexes:
                        batch_indices.append(index[kk])
                else:
                    select_indexes = self.no_index(index, i)
                    if not select_indexes:
                        # Only one image for this identity
                        ind_indexes = [0] * (self.num_instances - 1)
                    elif len(select_indexes) >= self.num_instances:
                        ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
                    else:
                        ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)

                    for kk in ind_indexes:
                        batch_indices.append(index[kk])

            yield from batch_indices
            batch_count += 1

        self.length = batch_count * self.batch_size