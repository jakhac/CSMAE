"""
Dataloader and Datamodule for BigEarthNet dataset. Files can be requested by contacting
the author.
Original Paper of Image Data:
https://arxiv.org/abs/2105.07921
https://bigearth.net/
"""

import csv
import os
import socket
from typing import Optional
import numpy as np

from torch.utils.data import Dataset

from src.bigearthnet_dataset.BEN_lmdb_utils import ben19_list_to_onehot
from src.bigearthnet_dataset.BEN_lmdb_utils import BENLMDBReader
from src.utils import Messages


class BENDataSet(Dataset):
    def __init__(
        self,
        root_dir="./",
        split_dir="./",
        split: Optional[str] = None,
        transform=None,
        max_img_idx=None,
        img_size=(12, 120, 120),
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split_dir = split_dir
        self.split = split
        self.lmdb_dir = os.path.join(self.root_dir, "BigEarthNetEncoded.lmdb")
        self.transform = transform
        self.image_size = img_size

        assert img_size[0] in [2, 3, 4, 10, 12], (
            "Image Channels have to be "
            "2 (Sentinel-1), "
            "3 (RGB), "
            "4 (10m Sentinel-2), "
            "10 (10m + 20m Sentinel-2) or "
            "12 (10m + 20m Sentinel-2 + 10m Sentinel-1) "
            "but was " + f"{img_size[0]}"
        )
        self.read_channels = img_size[0]

        print(f"LMDB {self.lmdb_dir}")
        print(f"Loading BEN data for {self.split}...")
        if split is not None:
            with open(os.path.join(self.split_dir, f"{self.split}.csv")) as f:
                reader = csv.reader(f)
                patches = list(reader)
        else:
            splits = ["train", "val", "test"]
            patches = []
            for s in splits:
                with open(os.path.join(self.split_dir, s + ".csv")) as f:
                    reader = csv.reader(f)
                    patches += list(reader)

        # lines get read as arrays -> flatten to one dimension
        self.patches = [x[0] for x in patches]
        
        # sort list for reproducibility
        self.patches.sort()
        print(f"    {len(self.patches)} patches indexed")
        if (
            max_img_idx is not None
            and max_img_idx < len(self.patches)
            and max_img_idx != -1
        ):
            self.patches = self.patches[:max_img_idx]

        print(f"    {len(self.patches)} filtered patches indexed")
        self.BENLoader = BENLMDBReader(
            lmdb_dir=self.lmdb_dir,
            label_type="new",
            image_size=self.image_size,
            bands=self.image_size[0],
        )

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        key = self.patches[idx]

        # get (& write) image from (& to) LMDB
        # get image from database
        # we have to copy, as the image in imdb is not writeable,
        # which is a problem in .to_tensor()
        img, labels = self.BENLoader[key]

        img = np.array(img.numpy())
        img = np.transpose(img,(2,1,0))

        if img is None:
            print(f"Cannot load {key} from database")
            raise ValueError
        if self.transform:
            img = self.transform(img)

        label_list = labels
        labels = ben19_list_to_onehot(labels)

        assert sum(labels) == len(set(label_list)), f"Label creation failed for {key}"
        
        return (idx, img, labels)
    