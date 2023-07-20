import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision
import torch

class ImageList(torchvision.datasets.VisionDataset):
    def __init__(self, config_path, root=None, transform=None, target_transform=None):
        if root is None:
            root = os.path.split(config_path)[0]
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        with open(config_path, 'r') as file:
            lines = file.readlines()
        self.samples = [line.strip().split(' ') for line in lines]
        self.samples = [(os.path.join(root, path), int(target)) for path, target in self.samples]
        self.imgs = self.samples
        self.paths, self.targets = list(zip(*self.imgs))
        self.classes = np.unique(self.targets)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, "rb") as file:
            img = Image.open(file)
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class ClassBalancedRandomSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(self, targets, num_samples=None, replacement=True, generator=None):
        if num_samples is None:
            num_samples = len(targets)
        targets = np.asarray(targets)
        classes, num_classes = np.unique(targets, return_counts=True)
        weight_classes = 1 / num_classes
        weight_classes = weight_classes / np.sum(weight_classes)
        targets = targets.reshape((-1, 1))
        classes = classes.reshape((1, -1))
        weights = np.sum((targets == classes) * weight_classes, axis=-1)
        super().__init__(weights, num_samples,
                         replacement=replacement, generator=generator)

def imagelist_to_imagefolder(img_list, tmp_dir='.tmp', tmp_class_dir='.class1', tmp_jpg='.tmp.jpg'):
    """ solve Broken pipe issue in windows when num_workers != 0"""
    # create fake image folder
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, tmp_class_dir), exist_ok=True)
    Image.new('RGB', (16, 16)).save(os.path.join(tmp_dir, tmp_class_dir, tmp_jpg))

    # create image folder instance
    img_folder = torchvision.datasets.ImageFolder(tmp_dir,
                                                  transform=img_list.transform,
                                                  target_transform=img_list.target_transform)
    # set attributes of img_list to img_folder
    img_folder.imgs = img_list.imgs
    img_folder.samples = img_list.samples
    img_folder.paths = img_list.paths
    img_folder.targets = img_list.targets
    img_folder.classes = img_list.classes
    return img_folder

def train_test_split_dataset(ds, train_size=0.7, seed=None):
    idx = np.arange(len(ds))
    train_idx, test_idx = train_test_split(idx, train_size=train_size, random_state=seed, shuffle=True, stratify=ds.targets)
    train_ds = torch.utils.data.Subset(ds, train_idx)
    test_ds = torch.utils.data.Subset(ds, test_idx)
    return train_ds, test_ds
