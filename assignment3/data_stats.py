from torch.utils.data import DataLoader
import torch
import data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = './dataset'

train_dataset = data.TrainDataset(root)
dataloader = DataLoader(train_dataset, batch_size=256, shuffle=False)

mean = 0.
std = 0.
nb_samples = 0.
for anchor in dataloader:
    data = anchor['image'].to(device)
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f'\rMean: {mean}')
print(f'\rStd: {std}')
