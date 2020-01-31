import matplotlib.pyplot as plt
import numpy as np
import cv2

mean = [0.1173, 0.0984, 0.0915]
std = [0.2281, 0.1765, 0.1486]

BATCH_SIZE = 32

def unnormalize(image):
    return image * std + mean


def visualize_triplet(anchor, puller, pusher, size = 4):
    # size = anchor['image'].size()[0]//4 + 6 # we are going to visualize only the first 4 images of the batch
    fig, ax = plt.subplots(size, 3, figsize=(50,50))

    for i in range(size):
        ax[i, 0].imshow(unnormalize(anchor['image'][i].numpy().transpose(1, 2, 0)))
        ax[i, 1].imshow(unnormalize(puller['image'][i].numpy().transpose(1, 2, 0)))
        ax[i, 2].imshow(unnormalize(pusher['image'][i].numpy().transpose(1, 2, 0)))
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        ax[i, 2].set_axis_off()
    fig.show()


def knn_to_dbdataset(descriptors_testdataset, descriptors_dbdataset):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.match(descriptors_testdataset, descriptors_dbdataset)

    return matches


def make_histogram(data):
    bins = np.zeros(4)
    for angular_diff in data:
        if angular_diff < 10:
            bins[0:4] += 1
        elif 10 < angular_diff < 20:
            bins[1:4] += 1
        elif 20 < angular_diff < 40:
            bins[2:4] += 1
        elif 40 < angular_diff < 180:
            bins[3] += 1

    return bins

def visualize_histogram(angular_diffs):
    bins = make_histogram(angular_diffs)

    bin_labels = ('<10', '<20', '<40', '<180')
    y_pos = np.arange(len(bin_labels))
    plt.bar(y_pos, bins.tolist(), align='center', alpha=0.5)
    plt.xticks(y_pos, bin_labels)
    for i in range(4):
        plt.text(y_pos[i], bins[i], str(bins[i]))
    plt.show()

def compute_angle(quaternion1, quaternion2):

    assert quaternion1.shape[0] == quaternion2.shape[0] == 4
    quaternion1 = quaternion1.numpy()
    quaternion2 = quaternion2.numpy()
    dot_res = np.minimum(1, quaternion1 @ quaternion2)
    dot_res = np.maximum(-1, dot_res)
    return 2 * np.rad2deg(np.arccos(np.abs(dot_res)).item())