import os.path
from torch.utils.data import DataLoader
import torchvision.transforms as T
import data
import utils

root = './dataset'

mean = [0.1173, 0.0984, 0.0915]
std = [0.2281, 0.1765, 0.1486]
transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

train_dataset = data.TrainDataset(root, transform=transform)  # Requires normalization
db_dataset = data.DbDataset(root, transform=transform)  # Requires normalization
dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for anchor in dataloader:
    puller_idx = db_dataset.get_puller(anchor['pose'], anchor['target'])
    pusher_idx = db_dataset.get_pusher(puller_idx, anchor['target'])
    puller = db_dataset.__getitem__(puller_idx, anchor['target'])
    pusher = db_dataset.__getitem__(pusher_idx, anchor['target'])
    utils.plot_triplet(anchor, puller, pusher)
