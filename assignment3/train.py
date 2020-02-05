import numpy as np
import torch
from datetime import datetime

import net
import utils
from loss import *
from torch.utils.tensorboard import SummaryWriter
import data_generator
import viz

np.seterr(all='raise')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = './model.pth'

# Writer will output to ./runs/ directory by default
# To visualize results write in terminal >> tensorboard --logdir=runs
# tensorflow must be install in environment
writer = SummaryWriter()
model = net.Net().to(device)

SAVE_CKP_EVERY = 1
UPDATE_LR_EVERY = 5

running_loss_history = []
running_acc_history = []


def print_loss_acc(DEBUG, running_loss, dg):
    if DEBUG:
        bins = viz.compute_histogram(model, dg, count_only=True)
        bins = np.array(bins, dtype=np.uint32)
        epoch_acc = float(bins[3]) / len(dg.test_dataset)
        running_acc_history.append(epoch_acc)
    epoch_loss = running_loss / len(dg.train_loader)
    running_loss_history.append(epoch_loss)
    if DEBUG:
        print("Epoch avg. loss: {:.4f}, Epoch acc: {}/{} = {:.4f}% ([<10]: {}, [<20], {}, [<40]: {}".format(
            epoch_loss, bins[3], len(dg.test_dataset), epoch_acc * 100, bins[0], bins[1], bins[2]))
    else:
        print("Epoch avg. loss: {:.4f}".format(epoch_loss))


def write_log(iter_i, log_embedding, dg):
    print("Reached iter {}, storing histogram...".format(iter_i))
    test_descriptors = net.compute_descriptor(model, dg.test_loader)
    hist_bins, hist_fig = viz.compute_histogram(model, dg, test_descriptors=test_descriptors)
    writer.add_scalar("Accuracy", hist_bins[3] / len(dg.test_dataset))
    writer.add_figure("Histogram of angular differences between TEST set and DB set",
                      hist_fig, global_step=iter_i)
    if log_embedding:
        writer.add_embedding(test_descriptors, metadata=dg.test_labels, tag='test_descriptors')
        writer.add_embedding(test_descriptors, metadata=dg.test_labels, label_img=dg.test_images,
                             tag='test_descriptors (w/ images)')


##############################################################
#                         TRAINING                           #
##############################################################
def run(dg, batch_size=128, num_epochs=5, lr=0.001):
    if dg is None:
        dg = data_generator.DataGenerator(root='./dataset', batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0, 0))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)

    total_iters = -(-len(dg.train_dataset) // batch_size) * num_epochs
    print("NUM_EPOCHS = {}, BATCH_SIZE = {}, len(train_set) = {} "
          "--> #Iterations = {}\n".format(num_epochs, batch_size, len(dg.train_dataset), total_iters))
    start_time = datetime.now()
    iter_i = 1
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}')
        running_loss = 0.0

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
            iter_i += 1

            if iter_i % 10 == 0:
                writer.add_scalar("Loss", loss.item(), iter_i)
            if iter_i % 1000 == 0:
                write_log(iter_i, log_embedding=True, dg=dg)

        else:
            # Note: set DEBUG=True to see classification acc. after every epoch but comes with the cost of
            # computing the histogram, which takes up some time and therefore leading to a longer training time
            DEBUG = True
            print_loss_acc(DEBUG, running_loss, dg)
            if total_iters < 1000 and epoch % 5 == 0:
                log_embedding = True if epoch % 20 == 0 else False
                write_log(iter_i, log_embedding, dg)

            if (epoch % SAVE_CKP_EVERY) == 0:
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                net.save_ckp(checkpoint, './models', epoch)

            if epoch % UPDATE_LR_EVERY == 0:
                scheduler.step()
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                    print(f'Learning rate updated to: {lr}')

    timeElapsed = datetime.now() - start_time
    print('Finished Training! Time elapsed (hh:mm:ss.ms) {}'.format(timeElapsed))
    print("\nHistory:")
    print(running_loss_history)
    print(running_acc_history)
    write_log(iter_i - 1, log_embedding=True, dg=dg)
    writer.close()


if __name__ == '__main__':
    batch_size = 32
    dg = data_generator.DataGenerator(root='./dataset', batch_size=batch_size)
    run(dg, batch_size=batch_size, num_epochs=50, lr=5e-4)
