import matplotlib.pyplot as plt
import numpy as np
import data_generator
import cv2
import torch


def train_denormalize(image):
    return np.clip(image * data_generator.train_std + data_generator.train_mean, 0, 1)


def db_denormalize(image):
    return np.clip(image * data_generator.db_std + data_generator.db_mean, 0, 1)

def visualize_triplet(anchor, puller, pusher, num_display = 4):
    fig, ax = plt.subplots(num_display, 3, figsize=(50, 50))

    for i in range(num_display):
        ax[i, 0].imshow(train_denormalize(anchor['image'][i].numpy().transpose(1, 2, 0)))
        ax[i, 1].imshow(db_denormalize(puller['image'][i].numpy().transpose(1, 2, 0)))
        ax[i, 2].imshow(db_denormalize(pusher['image'][i].numpy().transpose(1, 2, 0)))
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        ax[i, 2].set_axis_off()
    fig.show()


def convert_to_numpy_array(tensor1, tensor2):
    if torch.is_tensor(tensor1):
        tensor1 = tensor1.numpy()
    if torch.is_tensor(tensor2):
        tensor2 = tensor2.numpy()
    return tensor1, tensor2

def knn_to_dbdataset(test_descriptors, db_descriptors):
    test_descriptors, db_descriptors = convert_to_numpy_array(test_descriptors, db_descriptors)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.match(test_descriptors, db_descriptors)

    return matches


def make_histogram(data):
    bins = np.zeros(4)
    for angular_diff in data:
        assert 0 <= angular_diff and angular_diff <= 180, "Angualar difference out of range. (angular_diff = {.3f})".format(angular_diff)

        if angular_diff < 10:
            bins[0] += 1
        if angular_diff < 20:
            bins[1] += 1
        if angular_diff < 40:
            bins[2] += 1
        if angular_diff <= 180:
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
    quaternion1, quaternion2 = convert_to_numpy_array(quaternion1, quaternion2)
    dot_res = np.clip(quaternion1 @ quaternion2, -1, 1)
    return 2 * np.rad2deg(np.arccos(np.abs(dot_res)).item())

