# Separation of codes for data management gives better readablility and modularity

import torch
from torch.utils.data import Dataset  # Saves samples with labels
from torch.utils.data import DataLoader  # Provides iterables that contain data
from PIL import Image
import os
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


# MVTec
mvtec_root = os.path.join("data", "img", "MVTecAD")
mvtec_cls = ["tile"]


class MVTecDataset(Dataset):
    def __init__(
        self,
        img_size: int,
        apply_transform: bool = False,
        cls: str = "tile",
        phase: str = "train",
    ):
        self.cls = cls
        self.input_size = (img_size, img_size)
        self.apply_transform = apply_transform
        self.phase = phase
        self.img_dirs, self.gt_dirs, self.labels = self.load_folder()

        if apply_transform:
            self.transform_x = T.Compose(
                [
                    T.Resize(self.input_size, interpolation=InterpolationMode.LANCZOS),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            self.transform_mask = T.Compose(
                [
                    T.Resize(self.input_size, interpolation=InterpolationMode.NEAREST),
                    T.ToTensor(),
                ]
            )

    def load_folder(self):
        img_paths, gt_paths, labels = [], [], []
        img_dir = os.path.join(mvtec_root, self.cls, self.phase)
        gt_dir = os.path.join(mvtec_root, self.cls, "ground_truth")

        for defect in os.listdir(img_dir):
            defect_dir = os.path.join(img_dir, defect)
            for f in sorted(os.listdir(defect_dir)):
                img_paths.append(os.path.join(defect_dir, f))
                labels.append(0 if defect == "good" else 1)

                if defect == "good":
                    gt_paths.append(None)
                else:
                    mask_fname = os.path.splitext(f)[0] + "_mask.png"
                    gt_paths.append(os.path.join(gt_dir, defect, mask_fname))

        return img_paths, gt_paths, labels

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        img_path, gt_path, label = (
            self.img_dirs[idx],
            self.gt_dirs[idx],
            self.labels[idx],
        )

        img = Image.open(img_path).convert("RGB")
        if self.apply_transform:
            img = self.transform_x(img)

        if gt_path is None:
            gt = torch.zeros((1, self.input_size[0], self.input_size[1]))
        else:
            gt = Image.open(gt_path)
            if self.apply_transform:
                gt = self.transform_mask(gt)

        return img, gt, label
