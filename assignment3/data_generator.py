import numpy as np
import torch
import utils
from torchvision import transforms
from torch.utils.data import DataLoader
import datasets

mean = [0.2379, 0.2032, 0.1833]
std = [0.2102, 0.1828, 0.1662]


class DataGenerator():
    def __init__(self, root, batch_size=128):

        self.batch_size = batch_size


        self.transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=mean,std=std)
            ])

        # self.transform = None

        self.train_dataset = datasets.TRAIN(root, transform=self.transform)
        self.db_dataset = datasets.DB(root, transform=self.transform)
        self.test_dataset = datasets.TEST(root, transform=self.transform)

        self.database_dataset = datasets.DATABASE(root, transform=self.transform)
        self.database_loader = DataLoader(self.database_dataset, batch_size, shuffle=False)

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size = batch_size, shuffle=True)
        self.db_loader = DataLoader(dataset=self.db_dataset, batch_size = batch_size, shuffle=False)
        self.test_loader = DataLoader(dataset = self.test_dataset, batch_size = batch_size, shuffle=False)

        self.num_batches = len(self.train_loader)


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
        # utils.visualize_triplet(anchor, puller, pusher, size=10)

        # Create a tensor of zeros to reorganize the anchor,puller and pusher images
        # to have the shape stated on the homework :
        # anchor_1,puller_1,pusher_1,anchor_2,puller_2,pusher_2.....,anchor_batch_size,puller_batch_size,pusher_batch_size
        inputs = torch.zeros_like(anchor['image']).repeat(3, 1, 1, 1)
        inputs[0: self.batch_size * 3:3, :, :, :] = anchor['image']
        inputs[1: self.batch_size * 3:3, :, :, :] = puller['image']
        inputs[2: self.batch_size * 3:3, :, :, :] = pusher['image']

        return inputs
