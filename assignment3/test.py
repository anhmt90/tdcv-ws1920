import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import data
import model
import utils


def compute_histogram():
    testloader = DataLoader(test_dataset, batch_size*3, shuffle=False)
    dbloader = DataLoader(database_dataset, batch_size*3, shuffle=False)

    with torch.no_grad():
        output_test = np.concatenate([net(samples['image']).numpy() for j, samples in enumerate(testloader)])
        output_db = np.concatenate([net(samples['image']).numpy() for j, samples in enumerate(dbloader)])

        angular_diffs = []
        for match in utils.knn_to_dbdataset(output_test, output_db):
            m = test_dataset.__getitem__(match.queryIdx)
            n = database_dataset.__getitem__(match.trainIdx)
            if m['target'] == n['target']:
                angular_diffs.append(utils.compute_angle(m['pose'], n['pose']))

        utils.visualize_histogram(angular_diffs)


if __name__ == '__main__':
    root = './dataset'
    ckp_dir = 'models'
    ckp_file = 'checkpoint0.pt'
    ckp_path = os.path.join(ckp_dir, ckp_file)
    net = model.Net()
    net, _ = model.load_ckp(ckp_path, net)
    # mean and std from the train dataset
    mean = [0.1173, 0.0984, 0.0915]
    std = [0.2281, 0.1765, 0.1486]

    # Define transformations that will be apply to the images
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    test_dataset = data.TestDataset(root, transform=transform)  # Requires normalization
    database_dataset = data.DatabaseDataset(root, transform=transform)  # Requires normalization

    batch_size = 256
    compute_histogram()