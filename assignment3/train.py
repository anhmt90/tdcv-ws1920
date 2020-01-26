import os.path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import model
import data
import utils
from loss import *
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = './dataset'

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

batch_size = 64

# Dataloader to iterate through TRAIN SET
dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

# Create the model
net = model.Net()
net = net.to(device)

# Set up the optimizer
learning_rate = 1e-2
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0, 0))

num_epochs = 20

# TRAINING
for epoch in range(num_epochs):

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
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0
            writer.add_scalar('Loss', running_loss / 10, epoch * len(dataloader) + i)

print('Finished Training')
