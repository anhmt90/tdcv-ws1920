import os.path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import model
import data
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = './dataset'

mean = [0.1173, 0.0984, 0.0915]
std = [0.2281, 0.1765, 0.1486]
transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

batch_size=16
train_dataset = data.TrainDataset(root, transform=transform)  # Requires normalization
db_dataset = data.DbDataset(root, transform=transform)  # Requires normalization
dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

net = model.Net()
net = net.to(device)

for anchor in dataloader:
    puller_idx = db_dataset.get_puller(anchor['pose'], anchor['target'])
    pusher_idx = db_dataset.get_pusher(puller_idx)
    puller = db_dataset.__getitem__(puller_idx, anchor['target'])
    pusher = db_dataset.__getitem__(pusher_idx, anchor['target'])
    #utils.visualize_triplet(anchor, puller, pusher)
    x = torch.zeros_like(anchor['image']).repeat(3,1,1,1)
    x[0:batch_size*3:3,:,:,:] = anchor['image']
    x[1:batch_size * 3:3, :, :, :] = puller['image']
    x[2:batch_size * 3:3, :, :, :] = pusher['image']
