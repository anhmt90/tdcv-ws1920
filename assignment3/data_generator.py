import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import datasets
import viz
from matplotlib import pyplot as plt
import cv2



train_mean = [0.20495705, 0.17493474, 0.1553751 ]
train_std = [0.26818034, 0.22731194, 0.21109529]

db_mean = [0.11668851, 0.09778491, 0.09104513]
db_std = [0.24001887, 0.19127475, 0.17848527]

test_mean = [0.39331356, 0.33969042, 0.29303524]
test_std = [0.22380222, 0.20981997, 0.21034092]



def train_denormalize(image):
    return np.clip(image * train_std + train_mean, 0, 1)

def db_denormalize(image):
    return np.clip(image * db_std + db_mean, 0, 1)



class DataGenerator():
    def __init__(self, root, batch_size = 256):
        self.batch_size = batch_size

        train_transform = transforms.Compose([
            GaussianNoise(0, 15),
            transforms.ToTensor(),
            transforms.Normalize(mean=train_mean, std=train_std)
        ])

        db_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=db_mean, std=db_std)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=test_mean, std=test_std)
        ])

        # transform = None

        self.train_dataset = datasets.TRAIN(root, transform = train_transform)
        self.db_dataset = datasets.DB(root, transform = db_transform)
        self.test_dataset = datasets.TEST(root, transform = test_transform)

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size = batch_size, shuffle=True)
        self.db_loader = DataLoader(dataset=self.db_dataset, batch_size = batch_size, shuffle=False)
        self.test_loader = DataLoader(dataset = self.test_dataset, batch_size = batch_size, shuffle=False)

        self.num_batches = len(self.train_loader)

        classes = ['ape', 'benchvise', 'cam', 'cat', 'duck']
        self.test_labels = [classes[t] for t in self.test_dataset.targets]


    def make_batch(self, anchor):
        # From each image on a batch of anchor select the puller and pusher indexes
        puller_idx = self.db_dataset.get_puller_idx(anchor['pose'], anchor['target'])
        pusher_idx = self.db_dataset.get_pusher_idx(puller_idx)

        # Get a dict containing the images and poses for those indexes
        puller = self.db_dataset.get_triplet(puller_idx, anchor['target'])
        class_pusher = np.random.randint(0, 5, size = self.batch_size)  # Now pusher comes from any class of the db Dataset
        pusher = self.db_dataset.get_triplet(pusher_idx, class_pusher)

        # This one to take pusher from same class as anchor and puller
        # pusher = db_dataset.get_triplet(pusher_idx, anchor['target'])

        # If you want to visualize some images of the batch use this function
        # viz.visualize_triplet(anchor, puller, pusher, num_display=6)

        # Create a tensor of zeros to reorganize the anchor,puller and pusher images
        # to have the shape stated on the homework :
        # anchor_1,puller_1,pusher_1,anchor_2,puller_2,pusher_2.....,anchor_batch_size,puller_batch_size,pusher_batch_size
        inputs = torch.zeros_like(anchor['image']).repeat(3, 1, 1, 1)
        inputs[0: self.batch_size * 3:3, :, :, :] = anchor['image']
        inputs[1: self.batch_size * 3:3, :, :, :] = puller['image']
        inputs[2: self.batch_size * 3:3, :, :, :] = pusher['image']
        return inputs


class GaussianNoise(object):
    def __init__(self, mean=0., std=20):
        self.std = std
        self.mean = mean

    def __call__(self, img):
        is_tensor = torch.is_tensor(img)
        if not is_tensor:
            np_img = np.array(img)
            noise = np.zeros_like(np_img)
            mean = np.repeat(self.mean, 3)
            std = np.repeat(self.std, 3)
            cv2.randn(noise, mean, std)
            noisy = np_img + noise
            noisy = Image.fromarray(noisy.astype('uint8'), 'RGB')

            return noisy

        img = img + torch.randn(img.size()) * self.std + self.mean


        return img

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)