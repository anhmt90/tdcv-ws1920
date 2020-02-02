import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import datasets
import net
import utils
import data_generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_histogram():
    dg = data_generator.DataGenerator(root = './dataset')

    with torch.no_grad():
        output_test = torch.cat([model(test_input['image']) for j, test_input in enumerate(dg.test_loader)])
        output_db = torch.cat([model(db_input['image']) for j, db_input in enumerate(dg.db_loader)])

        output_test = output_test.cpu().numpy()
        output_db = output_db.cpu().numpy()

        angular_diffs = []
        for match in utils.knn_to_dbdataset(output_test, output_db):
            m = dg.test_dataset.__getitem__(match.queryIdx)
            n = dg.db_dataset.__getitem__(match.trainIdx)
            if m['target'] == n['target']:
                angular_diffs.append(utils.compute_angle(m['pose'], n['pose']))

        utils.visualize_histogram(angular_diffs)


root = './dataset'
ckp_dir = 'models'
ckp_file = 'checkpoint0.pt'
ckp_path = os.path.join(ckp_dir, ckp_file)
model = net.Net()
model, _ = net.load_ckp(ckp_path, model)

compute_histogram()