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

np.seterr(all='raise')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = './dataset'
PATH = './net.pth'
ckp_dir = 'models'  # Path of directory where checkpoints of model will be saved during training
save_every = 1
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

batch_size = 256

# Dataloader to iterate through TRAIN SET
dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
num_batches = len(dataloader)

# Create the model
net = model.Net()
net = net.to(device)
# net.load_state_dict(torch.load(PATH))

# Set up the optimizer
learning_rate = 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0, 0))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)
num_epochs = 50

# TRAINING
for epoch in range(num_epochs):
    print(f'Epoch {epoch}')
    running_loss = 0.0

    for batch_i, anchor in enumerate(dataloader):

        # From each image on a batch of anchor select the puller and pusher indexes
        puller_idx = db_dataset.get_puller(anchor['pose'], anchor['target'])
        pusher_idx = db_dataset.get_pusher(puller_idx)

        # Get a dict containing the images and poses for those indexes
        puller = db_dataset.get_triplet(puller_idx, anchor['target'])
        class_pusher = np.random.randint(0, 5, size=batch_size)  # Now pusher comes from any class of the db Dataset
        pusher = db_dataset.get_triplet(pusher_idx, class_pusher)

        # This one to take pusher from same class as anchor and puller
        # pusher = db_dataset.get_triplet(pusher_idx, anchor['target'])

        # If you want to visualize some images of the batch use this function
        # utils.visualize_triplet(anchor, puller, pusher, size=10)

        # Create a tensor of zeros to reorganize the anchor,puller and pusher images
        # to have the shape stated on the homework :
        # anchor_1,puller_1,pusher_1,anchor_2,puller_2,pusher_2.....,anchor_batch_size,puller_batch_size,pusher_batch_size
        inputs = torch.zeros_like(anchor['image']).repeat(3, 1, 1, 1)
        inputs[0:batch_size * 3:3, :, :, :] = anchor['image']
        inputs[1:batch_size * 3:3, :, :, :] = puller['image']
        inputs[2:batch_size * 3:3, :, :, :] = pusher['image']
        inputs = inputs.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = total_loss(outputs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        iteration = epoch * num_batches + batch_i
        if (iteration + 1) % 10 == 0:  # print every 10 batches
            print('iter: %d, loss: %.3f' % (iteration + 1, running_loss))
            writer.add_scalar('Loss', running_loss, iteration)
            running_loss = 0.0

        # if (iteration + 1) % 1000 == 0:
        #     writer.add_histogram('Histogram of ')

    if (epoch % save_every) == (save_every - 1):
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        model.save_ckp(checkpoint, ckp_dir, epoch)

    scheduler.step()

    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        print(f'Learning rate updated to: {lr}')


print('Finished Training')
# torch.save(net.state_dict(), PATH)
