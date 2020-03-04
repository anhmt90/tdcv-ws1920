import os
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 8)
        self.conv2 = nn.Conv2d(16, 7, 5)
        self.fc1 = nn.Linear(12 * 12 * 7, 256)
        self.fc2 = nn.Linear(256, 16)

    def forward(self, x):
        # assert (x.size()[0]%3) == 0, 'Batch size must be divisible by 3'
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 12 * 12 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def compute_descriptor(model, dataloader):
    with torch.no_grad():
        des = torch.cat([model(input['image'].to(device)) for _, input in enumerate(dataloader)])
    return des

def save_ckp(state, checkpoint_dir, epoch):
    f_path = os.path.join(checkpoint_dir, f'checkpoint{epoch}.pt')
    torch.save(state, f_path)


def load_ckp(checkpoint_fpath, model):
    checkpoint = torch.load(checkpoint_fpath, map_location = str(device))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model, checkpoint['epoch']