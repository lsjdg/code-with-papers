# Separation of codes for data management gives better readablility and modularity

import torch
from torch.utils.data import Dataset  # Saves samples with labels
from torch.utils.data import DataLoader  # Provides iterables that contain data


class MVTecDataset(Dataset):
    def __init__(self, img_dir: str, img_size: str, transforms: bool = False):
        self.img_dir = img_dir
        self.img_size = img_size
        self.transforms = transforms
