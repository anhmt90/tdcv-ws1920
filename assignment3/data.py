import os.path
import re
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.utils.data
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

class train_dataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (string): Path to dataset folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dir_fine = os.path.join(root, 'fine')
        self.dir_real = os.path.join(root, 'real')
        self.transform = transform
        self.imgs = []
        self.poses = []
        self.targets = []

        # First, go through fine folder and extract all images, class and poses in three different lists
        valid_images = [".jpeg", ".jpg", ".png"]
        c = 0
        for folder in sorted(os.listdir(self.dir_fine)):
            fidx = 0
            num_lines = sum(1 for _ in open(os.path.join(self.dir_fine, folder, 'poses.txt')))
            skip_idx = list(range(0, num_lines - 1, 2))
            pose_folder = pd.read_csv(os.path.join(self.dir_fine, folder, 'poses.txt'), sep=' ', header=None,
                                      skiprows=skip_idx)
            for img in sorted_aphanumeric(os.listdir(os.path.join(self.dir_fine, folder))):
                ext = os.path.splitext(img)[1]
                if ext.lower() not in valid_images:
                    continue
                self.imgs.append(os.path.join(self.dir_fine, folder, img))
                self.poses.append(pose_folder.iloc[fidx].tolist())
                self.targets.append(c)
                fidx += 1
            c += 1

        # Now, do the same for images in real folder belonging to training set (split in training_split.txt)
        split_list = pd.read_csv(os.path.join(self.dir_real, 'training_split.txt'))
        c = 0
        for folder in sorted(os.listdir(self.dir_real)):
            if (folder == 'training_split.txt') == True or (folder == '.DS_Store') == True:
                continue
            num_lines = sum(1 for _ in open(os.path.join(self.dir_real, folder, 'poses.txt')))
            skip_idx = list(range(0, num_lines - 1, 2))
            pose_folder = pd.read_csv(os.path.join(self.dir_real, folder, 'poses.txt'), sep=' ', header=None,
                                      skiprows=skip_idx)
            for img in sorted_aphanumeric(os.listdir(os.path.join(self.dir_real, folder))):
                file, ext = os.path.splitext(img)
                if ext.lower() not in valid_images:
                    continue
                if file[len('real'):] in split_list:  # check if image_id is included in training split
                    self.imgs.append(os.path.join(self.dir_real, folder, img))
                    self.poses.append(pose_folder.iloc[int(file[len('real'):])].tolist())
                    self.targets.append(c)
            c += 1
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        pose = torch.Tensor(self.poses[idx])
        target = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        train_sample = {'image': img, 'pose': pose, 'target': target}
        return train_sample


class db_dataset():
    def __init__(self, root, transform=None):
        """
        Args:
            root (string): Path to dataset folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dir = os.path.join(root, 'coarse')
        self.transform = transform
        self.per_class_list = []
        valid_images = [".jpeg", ".jpg", ".png"]
        for folder in sorted(os.listdir(self.dir)):
            imgs = []
            poses = []
            fidx = 0
            num_lines = sum(1 for _ in open(os.path.join(self.dir, folder, 'poses.txt')))
            skip_idx = list(range(0, num_lines - 1, 2))
            pose_folder = pd.read_csv(os.path.join(self.dir, folder, 'poses.txt'), sep=' ', header=None, skiprows=skip_idx)
            for img in sorted_aphanumeric(os.listdir(os.path.join(self.dir, folder))):
                ext = os.path.splitext(img)[1]
                if ext.lower() not in valid_images:
                    continue
                imgs.append(os.path.join(self.dir, folder, img))
                poses.append(pose_folder.iloc[fidx].tolist())
                fidx += 1
            self.per_class_list.append(pd.DataFrame(zip(imgs, poses), columns=['images', 'poses']))
        self.imgs_per_class = len(imgs)

    def get_puller(self, anchor_pose, anchor_class):
        poses = self.per_class_list[anchor_class]['poses'].tolist()
        dist = []
        for p in poses:
            #b = np.linalg.norm(p, ord=2, axis=0)
            dist.append(2 * np.arccos((np.dot(anchor_pose.numpy(), p))**2 -1).item())
        return np.argmin(dist)

    def get_pusher(self, puller_idx, anchor_class):
        while True:
            pusher_idx = np.random.randint(0, self.imgs_per_class)
            if pusher_idx != puller_idx:
                return pusher_idx


    def __len__(self):
        return self.imgs_per_class

    def __getitem__(self, idx, anchor_class):
        img = Image.open(self.per_class_list[anchor_class]['images'].iloc[idx])
        pose = torch.Tensor(self.per_class_list[anchor_class]['poses'].iloc[idx])
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        train_sample = {'image': img, 'pose': pose}
        return train_sample
