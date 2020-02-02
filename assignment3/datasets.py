import os.path
import re
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.utils.data
from torchvision import transforms as T
from torch.utils.data import Dataset

import utils

DEBUG = False


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


class TEST(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (string): Path to dataset folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dir = os.path.join(root, 'real')
        self.transform = transform
        self.imgs = []
        self.poses = []
        self.targets = []
        valid_images = [".jpeg", ".jpg", ".png"]
        split_list = pd.read_csv(os.path.join(self.dir, 'training_split.txt'), header=None).values.flatten().tolist()
        class_ = 0
        skip_one = 0

        for folder in sorted(os.listdir(self.dir)):
            if (folder == 'training_split.txt') == True or (folder == '.DS_Store') == True:
                continue
            num_lines = sum(1 for _ in open(os.path.join(self.dir, folder, 'poses.txt')))
            skip_idx = list(range(0, num_lines - 1, 2))
            pose_folder = pd.read_csv(os.path.join(self.dir, folder, 'poses.txt'), sep=' ', header=None,
                                      skiprows=skip_idx)
            for img in sorted_alphanumeric(os.listdir(os.path.join(self.dir, folder))):
                file, ext = os.path.splitext(img)
                if ext.lower() not in valid_images:
                    continue
                if skip_one == 0:  # skip an image from the test data set so that it is divisible by 3
                    skip_one += 1
                    continue
                img_id = int(file[len('real'):])
                if img_id not in split_list:  # check if image_id is included in training split
                    self.imgs.append(os.path.join(self.dir, folder, img))
                    self.poses.append(pose_folder.iloc[int(file[len('real'):])].tolist())
                    self.targets.append(class_)
            class_ += 1

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
        test_sample = {'image': img, 'pose': pose, 'target': target}
        return test_sample


class TRAIN(Dataset):
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
            for img in sorted_alphanumeric(os.listdir(os.path.join(self.dir_fine, folder))):
                #### DEBUG
                if DEBUG:
                    if len(self.targets) >= (50 * (c+1)):
                        break

                ext = os.path.splitext(img)[1]
                if ext.lower() not in valid_images:
                    continue
                self.imgs.append(os.path.join(self.dir_fine, folder, img))
                self.poses.append(pose_folder.iloc[fidx].tolist())
                self.targets.append(c)
                fidx += 1

            c += 1

        # Now, do the same for images in real folder belonging to training set (split in training_split.txt)
        split_list = pd.read_csv(os.path.join(self.dir_real, 'training_split.txt'), header=None).values.flatten().tolist()
        c = 0
        for folder in sorted(os.listdir(self.dir_real)):
            if (folder == 'training_split.txt') == True or (folder == '.DS_Store') == True:
                continue
            num_lines = sum(1 for _ in open(os.path.join(self.dir_real, folder, 'poses.txt')))
            skip_idx = list(range(0, num_lines - 1, 2))
            pose_folder = pd.read_csv(os.path.join(self.dir_real, folder, 'poses.txt'), sep=' ', header=None,
                                      skiprows=skip_idx)
            for img in sorted_alphanumeric(os.listdir(os.path.join(self.dir_real, folder))):
                #### DEBUG
                if DEBUG:
                    if len(self.targets) >= (50 * 5) + (50 * (c + 1)):
                        break

                file, ext = os.path.splitext(img)
                if ext.lower() not in valid_images:
                    continue
                img_id = int(file[len('real'):])
                if img_id in split_list:  # check if image_id is included in training split
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


class DB(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (string): Path to dataset folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dir = os.path.join(root, 'coarse')
        self.transform = transform

        self.imgs = []
        self.poses = []
        self.targets = []

        valid_images = [".jpeg", ".jpg", ".png"]

        class_ = 0
        for folder in sorted(os.listdir(self.dir)):
            imgs_ = []
            poses_ = []

            num_lines = sum(1 for _ in open(os.path.join(self.dir, folder, 'poses.txt')))
            skip_idx = list(range(0, num_lines - 1, 2))
            pose_folder = pd.read_csv(os.path.join(self.dir, folder, 'poses.txt'), sep=' ', header=None, skiprows=skip_idx)

            file_idx = 0
            for img in sorted_alphanumeric(os.listdir(os.path.join(self.dir, folder))):
                ext = os.path.splitext(img)[1]
                if ext.lower() not in valid_images:
                    continue
                self.imgs.append(os.path.join(self.dir, folder, img))
                imgs_.append(os.path.join(self.dir, folder, img))

                self.poses.append(pose_folder.iloc[file_idx].tolist())
                poses_.append(pose_folder.iloc[file_idx].tolist())

                self.targets.append(class_)
                file_idx += 1
            class_ += 1
        self.imgs_per_class = len(self.targets) // class_

    def get_puller_idx(self, anchor_pose, anchor_class):
        indices = []
        for anchor_c, anchor_p in zip(anchor_class, anchor_pose):
            start, end = self.get_slice(anchor_c)
            poses = np.array(self.poses)[start:end, :]
            dist = []
            for p in poses:
                dist.append(utils.compute_angle(anchor_p, p))
            indices.append(np.argmin(dist))
        return indices


    def get_pusher_idx(self, puller_idx):
        indices = []
        for i in puller_idx:
            while True:
                pusher_idx = np.random.randint(0, self.imgs_per_class)
                if pusher_idx != i:
                    indices.append(pusher_idx)
                    break
        return indices

    def get_triplet(self, idx, anchor_class):
        image_list, pose_list = [], []
        for i, anchor_c in zip(idx, anchor_class):
            start, end = self.get_slice(anchor_c)
            img = Image.open(np.array(self.imgs)[start:end][i])
            pose = torch.Tensor(np.array(self.poses)[start:end][i])
            if self.transform:
                img = self.transform(img)
            else:
                img = T.ToTensor()(img)
            image_list.append(img.unsqueeze(0))
            pose_list.append(pose.unsqueeze(0))
        train_sample = {'image': torch.cat(image_list, dim=0), 'pose': torch.cat(pose_list, dim=0)}
        return train_sample

    def get_slice(self, anchor_c):
        start = anchor_c.item() * self.imgs_per_class
        end = start + self.imgs_per_class
        return start, end

    def __len__(self):
        return len(self.targets)

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
