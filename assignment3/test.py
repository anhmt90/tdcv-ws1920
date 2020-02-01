import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import datasets
import net
import utils
from utils import BATCH_SIZE


def compute_histogram():
    testloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)
    dbloader = DataLoader(database_dataset, BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        output_test = np.concatenate([model(samples['image']).numpy() for j, samples in enumerate(testloader)])
        output_db = np.concatenate([model(samples['image']).numpy() for j, samples in enumerate(dbloader)])

        angular_diffs = []
        for match in utils.knn_to_dbdataset(output_test, output_db):
            m = test_dataset.__getitem__(match.queryIdx)
            n = database_dataset.__getitem__(match.trainIdx)
            if m['target'] == n['target']:
                angular_diffs.append(utils.compute_angle(m['pose'], n['pose']))

        utils.visualize_histogram(angular_diffs)


root = './dataset'
ckp_dir = 'models'
ckp_file = 'checkpoint4.pt'
ckp_path = os.path.join(ckp_dir, ckp_file)
model = net.Net()
model, _ = net.load_ckp(ckp_path, model)
# mean and std from the train dataset
# mean = [0.1173, 0.0984, 0.0915]
# std = [0.2281, 0.1765, 0.1486]

# Define transformations that will be apply to the images
transform = T.Compose([
                       T.ToTensor(),
                       T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ])

test_dataset = datasets.TEST(root, transform=transform)  # Requires normalization
database_dataset = datasets.DB(root, transform=transform)  # Requires normalization

compute_histogram()