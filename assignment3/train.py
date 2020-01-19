import os.path
from torch.utils.data import DataLoader
import data

root = './dataset'

train_dataset = data.train_dataset(root)  # Requires normalization
db_dataset = data.db_dataset(root)  # Requires normalization
dataloader = DataLoader(train_dataset, shuffle=True)

for anchor in dataloader:
    puller_idx = db_dataset.get_puller(anchor['pose'], anchor['target'])
    pusher_idx = db_dataset.get_pusher(puller_idx, anchor['target'])
    puller = db_dataset.__getitem__(puller_idx, anchor['target'])
    pusher = db_dataset.__getitem__(pusher_idx, anchor['target'])
