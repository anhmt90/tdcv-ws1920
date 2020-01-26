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


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


class TestDataset(Dataset):
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
        c = 0
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
                img_id = int(file[len('real'):])
                if img_id not in split_list:  # check if image_id is included in training split
                    self.imgs.append(os.path.join(self.dir, folder, img))
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
        test_sample = {'image': img, 'pose': pose, 'target': target}
        return test_sample


class TrainDataset(Dataset):
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


class DbDataset(Dataset):
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
            for img in sorted_alphanumeric(os.listdir(os.path.join(self.dir, folder))):
                ext = os.path.splitext(img)[1]
                if ext.lower() not in valid_images:
                    continue
                imgs.append(os.path.join(self.dir, folder, img))
                poses.append(pose_folder.iloc[fidx].tolist())
                fidx += 1
            self.per_class_list.append(pd.DataFrame(zip(imgs, poses), columns=['images', 'poses']))
        self.imgs_per_class = len(imgs)

    def get_puller(self, anchor_pose, anchor_class):
        indices = []
        for anchor_c, anchor_p in zip(anchor_class, anchor_pose):
            poses = self.per_class_list[anchor_c]['poses'].tolist()
            dist = []
            for p in poses:
                dist.append(2 * np.arccos((np.dot(anchor_p.numpy(), p))**2 -1).item())
            indices.append(np.argmin(dist))
        return indices

    def get_pusher(self, puller_idx):
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
            img = Image.open(self.per_class_list[anchor_c]['images'].iloc[i])
            pose = torch.Tensor(self.per_class_list[anchor_c]['poses'].iloc[i])
            if self.transform:
                img = self.transform(img)
            else:
                img = T.ToTensor()(img)
            image_list.append(img.unsqueeze(0))
            pose_list.append(pose.unsqueeze(0))
        train_sample = {'image': torch.cat(image_list, dim=0), 'pose': torch.cat(pose_list, dim=0)}
        return train_sample

    def __len__(self):
        return self.imgs_per_class

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