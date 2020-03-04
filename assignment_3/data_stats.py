from torch.utils.data import DataLoader
import torch
import numpy as np
import datasets
import data_generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# mean = 0.0
# std = 0.0
# nb_samples = 0.
# for anchor in dg.db_loader:
#     data = anchor['image'].to(device)
#     batch_samples = data.size(0)
#     data = data.view(data.size(0), data.size(1), -1)
#     mean += data.mean(2).sum(0)
#     std += data.std(2).sum(0)
#     nb_samples += batch_samples
#
# mean /= nb_samples
# std /= nb_samples
#
# print(f'\rMean: {mean}')
# print(f'\rStd: {std}')

def get_real_stats(dataset):
    all_data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    all_samples_mean = 0.0
    all_samples_std = 0.0
    for all_data in all_data_loader:
        all_data = torch.cat([samples.unsqueeze(0) for j, samples in enumerate(all_data['image'])])
        print(all_data.shape)
        all_samples_mean = np.mean(all_data.numpy(), axis=(0, 2, 3))
        all_samples_std = np.std(all_data.numpy(), axis=(0, 2, 3))
    return all_samples_mean, all_samples_std


def get_approx_stats(dataset, batch_size = 128):
    per_batch_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
    sample = dataset.__getitem__(0)['image']
    pixels_count = sample.shape[1] * sample.shape[2]
    all_batches_sum = 0.0
    all_batches_sum_sq = 0.0
    for batch in per_batch_loader:
        whole_batch = torch.cat([samples.unsqueeze(0) for j, samples in enumerate(batch['image'])])
        batch_sum = np.sum(whole_batch.numpy(), axis=(0, 2, 3))
        all_batches_sum += batch_sum

        batch_sum_sq = np.sum(whole_batch.numpy() ** 2, axis=(0, 2, 3))
        all_batches_sum_sq += batch_sum_sq

    samples_mean = all_batches_sum / (len(dataset) * pixels_count)
    samples_mean = np.atleast_1d(samples_mean)
    samples_squared_mean = all_batches_sum_sq / (len(dataset) * pixels_count)
    samples_std = np.sqrt(samples_squared_mean - samples_mean ** 2)
    samples_std = np.atleast_1d(samples_std)
    return samples_mean, samples_std

def test_approx_mean_std_whole_dataset(dataset):
    '''
        Sources:
            https://forums.fast.ai/t/image-normalization-in-pytorch/7534/10?u=anhmt
            https://github.com/labdmitriy/otus_ml/blob/master/Lesson_1/Homework/homework_transformer.ipynb
    '''
    print("ALL SAMPLES")
    all_samples_mean, all_samples_std = get_real_stats(dataset)
    print(all_samples_mean, all_samples_std)

    print("BATCH SAMPLES")
    samples_mean, samples_std = get_approx_stats(dataset)
    print(samples_mean, samples_std)


# test_approx_mean_std_whole_dataset(dg.train_dataset)

dg = data_generator.DataGenerator(root='./dataset')
def run():
    mean, std = get_real_stats(dg.train_dataset)
    print("Stats of TRAIN set")
    print(mean, std)

    mean, std = get_real_stats(dg.db_dataset)
    print("\nStats of DB set")
    print(mean, std)

    mean, std = get_real_stats(dg.test_dataset)
    print("\nStats of TEST set")
    print(mean, std)

####################### NOTE ############################
# Set transform=None in data_loader before running this #
#########################################################
run()