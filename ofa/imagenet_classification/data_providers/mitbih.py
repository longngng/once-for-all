import warnings
import os
import math
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import pandas as pd

from .base_provider import DataProvider
from ofa.utils.my_dataloader import MyRandomResizedCrop, MyDistributedSampler

__all__ = ["ECGDataProvider"]


class ECGDataset(Dataset):
    def __init__(self, file_path, dims=2):
        super(Dataset, self).__init__()
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path, header=None)
        self.dims = dims

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index].to_numpy()
        X = np.expand_dims(row[:-1], axis=0)

        if self.dims == 2:
            X = np.expand_dims(X, axis=0)
        X = X.astype("float")
        Y = int(row[-1])

        return X, Y


class ECGDataProvider(DataProvider):
    DEFAULT_PATH = "dataset/271022"

    def __init__(
        self,
        save_path=None,
        train_batch_size=128,
        test_batch_size=16,
        valid_size=None,
        n_worker=32,
        image_size=187,
        num_replicas=None,
        rank=None,
    ):

        warnings.filterwarnings("ignore")
        self._save_path = save_path

        self.image_size = image_size  # int or list of int

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            raise NotImplementedError
        else:
            self.active_img_size = self.image_size
            train_loader_class = torch.utils.data.DataLoader

        train_dataset = self.train_dataset()

        if valid_size is not None:
            raise NotImplementedError
        else:
            if num_replicas is not None:
                raise NotImplementedError
            else:
                self.train = train_loader_class(
                    train_dataset,
                    batch_size=train_batch_size,
                    shuffle=True,
                    num_workers=n_worker,
                    pin_memory=True,
                )
            self.valid = None

        test_dataset = self.test_dataset()
        if num_replicas is not None:
            raise NotImplementedError
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=True,
            )

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return "mit-bih"

    @property
    def data_shape(self):
        return 1, 1, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        return 5

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser("~/dataset/mitbih")
        return self._save_path

    def train_dataset(self):
        return ECGDataset(self.train_path)

    def test_dataset(self):
        return ECGDataset(self.valid_path)

    @property
    def train_path(self):
        return os.path.join(self.save_path, "mitbih_train.csv")

    @property
    def valid_path(self):
        return os.path.join(self.save_path, "mitbih_val.csv")

    @property
    def normalize(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def build_train_transform(self, image_size=None, print_log=True):
        raise NotImplementedError

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = self.active_img_size

    def build_sub_train_loader(
        self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None
    ):
        # used for resetting BN running statistics
        if self.__dict__.get("sub_train_%d" % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            n_samples = len(self.train.dataset)
            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()

            new_train_dataset = self.train_dataset()
            chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                raise NotImplementedError
            else:
                sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    chosen_indexes
                )
            sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset,
                batch_size=batch_size,
                sampler=sub_sampler,
                num_workers=num_worker,
                pin_memory=True,
            )
            self.__dict__["sub_train_%d" % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__["sub_train_%d" % self.active_img_size].append(
                    (images, labels)
                )
        return self.__dict__["sub_train_%d" % self.active_img_size]
