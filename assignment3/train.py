import os.path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import model
import data
import utils
import numpy as np
from loss import *
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = './dataset'
PATH = './net.pth'

# Writer will output to ./runs/ directory by default
# To visualize results write in terminal tensorboard --logdir=runs
# tensorflow must be install in environment
writer = SummaryWriter()

# mean and std from the train dataset
mean = [0.1173, 0.0984, 0.0915]
std = [0.2281, 0.1765, 0.1486]

# Define transformations that will be apply to the images
transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

# Create the three different datasets
train_dataset = data.TrainDataset(root, transform=transform)  # Requires normalization
db_dataset = data.DbDataset(root, transform=transform)  # Requires normalization
test_dataset = data.TestDataset(root, transform=transform)  # Requires normalization
database_dataset = data.DatabaseDataset(root, transform=transform)  # Requires normalization

batch_size = 64

# Dataloader to iterate through TRAIN SET
dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

# Create the model
net = model.Net()
net = net.to(device)
# net.load_state_dict(torch.load(PATH))


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
                angular_diffs.append(2 * np.arccos((np.dot(m['pose'].numpy(), n['pose'].numpy())) ** 2 - 1).item())

        utils.visualize_histogram(angular_diffs)


# Set up the optimizer
learning_rate = 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0, 0))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)
num_epochs = 20

# TRAINING
for epoch in range(num_epochs):
    print(f'Epoch {epoch}')
    running_loss = 0.0

    for i, anchor in enumerate(dataloader):

        # From each image on a batch of anchor select the puller and pusher indexes
        puller_idx = db_dataset.get_puller(anchor['pose'], anchor['target'])
        pusher_idx = db_dataset.get_pusher(puller_idx) # pusher belongs to the same class but has different pose to puller

        # Get a dict containing the images and poses for those indexes
        puller = db_dataset.get_triplet(puller_idx, anchor['target'])
        pusher = db_dataset.get_triplet(pusher_idx, anchor['target'])

        # If you want to visualize some images of the batch use this function
        # utils.visualize_triplet(anchor, puller, pusher)

        # Create a tensor of zeros to reorganize the anchor,puller and pusher images
        # to have the shape stated on the homework :
        # anchor_1,puller_1,pusher_1,anchor_2,puller_2,pusher_2.....,anchor_batch_size,puller_batch_size,pusher_batch_size
        x = torch.zeros_like(anchor['image']).repeat(3, 1, 1, 1)
        x[0:batch_size*3:3,:,:,:] = anchor['image']
        x[1:batch_size * 3:3, :, :, :] = puller['image']
        x[2:batch_size * 3:3, :, :, :] = pusher['image']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(x)
        loss = total_loss(outputs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # print every 10 batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch, i + 1, running_loss))
            writer.add_scalar('Loss', running_loss / 10, epoch * len(dataloader) + i)
            running_loss = 0.0

        # if (epoch+1)*(i+1) % 1000 == 0:
        #    compute_histogram()

    scheduler.step()
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        print(f'Learning rate updated to: {lr}')


print('Finished Training')
# torch.save(net.state_dict(), PATH)
