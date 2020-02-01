from torch.utils.data import DataLoader
import torch
import datasets
import data_generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = './dataset'

dg = data_generator.DataGenerator(root = './dataset')

mean = 0.0
std = 0.0
nb_samples = 0.
for anchor in dg.train_loader:
    data = anchor['image'].to(device)
    batch_samples = data.size(0)
    data = data.view(data.size(0), data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f'\rMean: {mean}')
print(f'\rStd: {std}')
