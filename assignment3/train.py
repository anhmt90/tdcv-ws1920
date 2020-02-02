import net
import utils
import numpy as np
from loss import *
from torch.utils.tensorboard import SummaryWriter
import data_generator

np.seterr(all='raise')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = './model.pth'
ckp_dir = 'models'  # Path of directory where checkpoints of model will be saved during training
save_every = 1
# Writer will output to ./runs/ directory by default
# To visualize results write in terminal tensorboard --logdir=runs
# tensorflow must be install in environment
writer = SummaryWriter()

# mean and std from the train dataset
mean = [0.1173, 0.0984, 0.0915]
std = [0.2281, 0.1765, 0.1486]

# Create the model
model = net.Net()
model = model.to(device)

# net.load_state_dict(torch.load(PATH))

# Set up the optimizer
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0, 0))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)

num_epochs = 15

dg = data_generator.DataGenerator(root = './dataset')

# TRAINING
running_loss_history = []
running_acc_history = []

for epoch in range(num_epochs):
    print(f'Epoch {epoch}')
    running_loss = 0.0
    running_corrects = 0

    for batch_i, anchor in enumerate(dg.train_loader):
        inputs = dg.make_batch(anchor)
        inputs = inputs.to(device)

        # forward + backward + optimize
        outputs = model(inputs)
        loss = total_loss(outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # iteration = epoch * dg.num_batches + batch_i
        # if (iteration + 1) % 10 == 0:  # print every 10 batches
        #     print('iter: %d, loss: %.3f' % (iteration + 1, running_loss))
        #     writer.add_scalar('Loss', running_loss, iteration)
        #     running_loss = 0.0

    else:
        with torch.no_grad():
            output_test = torch.cat([model(test_input['image'].to(device)) for j, test_input in enumerate(dg.test_loader)])
            output_db = torch.cat([model(db_input['image'].to(device)) for j, db_input in enumerate(dg.db_loader)])

            output_test = output_test.cpu().numpy()
            output_db = output_db.cpu().numpy()

            angular_diffs = []
            for match in utils.knn_to_dbdataset(output_test, output_db):
                m = dg.test_dataset.__getitem__(match.queryIdx)
                n = dg.db_dataset.__getitem__(match.trainIdx)
                if m['target'] == n['target']:
                    # angular_diffs.append(utils.compute_angle(m['pose'], n['pose']))
                    running_corrects += 1

        epoch_loss = running_loss / len(dg.train_loader)
        running_loss_history.append(epoch_loss)

        epoch_acc = float(running_corrects) / dg.test_dataset.__len__()
        running_acc_history.append(epoch_acc)
        print("Epoch avg. loss: {:.4f}, Epoch acc: {}/{}".format(epoch_loss, running_corrects, dg.test_dataset.__len__()))

        # if (iteration + 1) % 1000 == 0:
        #     writer.add_histogram('Histogram of ')

    if (epoch % save_every) == (save_every - 1):
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        net.save_ckp(checkpoint, ckp_dir, epoch)

    if (epoch % 5) == 4:
        scheduler.step()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            print(f'Learning rate updated to: {lr}')

print('Finished Training')
# torch.save(net.state_dict(), PATH)
print("\nHistory:")
print(running_loss_history)
print(running_acc_history)