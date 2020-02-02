import numpy as np
import torch
import net
import utils
from loss import *
from torch.utils.tensorboard import SummaryWriter
import data_generator

np.seterr(all='raise')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = './model.pth'

# Writer will output to ./runs/ directory by default
# To visualize results write in terminal >> tensorboard --logdir=runs
# tensorflow must be install in environment
writer = SummaryWriter()
model = net.Net().to(device)
# net.load_state_dict(torch.load(PATH))

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0, 0))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)

NUM_EPOCHS = 15
BATCH_SIZE = 64
dg = data_generator.DataGenerator(root='./dataset', batch_size=BATCH_SIZE)

running_loss_history = []
running_acc_history = []


def compute_histogram(logging=True):
    true_positives = 0
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
                if (logging):
                    angular_diffs.append(utils.compute_angle(m['pose'], n['pose']))
                else:
                    true_positives += 1

    if (logging):
        return angular_diffs
    return true_positives


def print_loss_acc(DEBUG, running_loss):
    if DEBUG:
        running_corrects = compute_histogram(logging=False)
        epoch_acc = float(running_corrects) / len(dg.test_dataset)
        running_acc_history.append(epoch_acc)
    epoch_loss = running_loss / len(dg.train_loader)
    running_loss_history.append(epoch_loss)
    if DEBUG:
        print("Epoch avg. loss: {:.4f}, Epoch acc: {}/{} = {:.4f}%".format(epoch_loss, running_corrects,
                                                                           len(dg.test_dataset), epoch_acc * 100))
    else:
        print("Epoch avg. loss: {:.4f}".format(epoch_loss))


##############################################################
#                         TRAINING                           #
##############################################################
def train():
    print("NUM_EPOCHS = {}, BATCH_SIZE = {}, len(train_set) = {} "
          "--> #Iterations = {}\n".format(NUM_EPOCHS, BATCH_SIZE, len(dg.train_dataset),
                                        -(-len(dg.train_dataset) // BATCH_SIZE) * NUM_EPOCHS))
    iter_i = 1
    for epoch in range(1, NUM_EPOCHS + 1):
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
                print("Reached iter {}, storing histogram...")
                angular_diffs = compute_histogram()
                histogram_bins = utils.make_histogram(angular_diffs)
                writer.add_scalars("Accuracy",
                                   {'true_positives': '{}/{}'.format(histogram_bins[3], len(dg.test_dataset)),
                                    'percentage:': float(histogram_bins[3]) / len(dg.test_dataset)})
                writer.add_histogram("Histogram of angular differences between TEST set and DB set",
                                     histogram_bins, global_step=iter_i)

        else:
            # Note: set DEBUG=True to see classification acc. after every epoch but comes with the cost of
            # computing the histogram, which takes up some time and therefore leading to a longer training time
            DEBUG = True
            print_loss_acc(DEBUG, running_loss)

            SAVE_CKP_EVERY = 1
            if (epoch % SAVE_CKP_EVERY) == 0:
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                net.save_ckp(checkpoint, './models', epoch)

            UPDATE_LR_EVERY = 5
            if epoch % UPDATE_LR_EVERY == 0:
                scheduler.step()
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                    print(f'Learning rate updated to: {lr}')

    print('Finished Training')
    # torch.save(net.state_dict(), PATH)
    print("\nHistory:")
    print(running_loss_history)
    print(running_acc_history)


train()
writer.close()
