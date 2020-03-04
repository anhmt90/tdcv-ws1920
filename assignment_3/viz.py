import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import utils
import net
import data_generator as dg_

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def visualize_triplet(anchor, puller, pusher, num_display=4):
    fig, ax = plt.subplots(num_display, 3, figsize=(50, 50))

    for i in range(num_display):
        ax[i, 0].imshow(dg_.train_denormalize(anchor['image'][i].numpy().transpose(1, 2, 0)))
        ax[i, 1].imshow(dg_.db_denormalize(puller['image'][i].numpy().transpose(1, 2, 0)))
        ax[i, 2].imshow(dg_.db_denormalize(pusher['image'][i].numpy().transpose(1, 2, 0)))
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        ax[i, 2].set_axis_off()
    fig.show()


def get_histogram_bins(data):
    bins = np.zeros(4)
    for angular_diff in data:
        assert 0 <= angular_diff and angular_diff <= 180, "Angualar difference out of range. (angular_diff = {.3f})".format(
            angular_diff)

        if angular_diff < 10:
            bins[0] += 1
        if angular_diff < 20:
            bins[1] += 1
        if angular_diff < 40:
            bins[2] += 1
        if angular_diff <= 180:
            bins[3] += 1
    return bins


def get_histogram_plot(bins):
    bin_labels = ('<10', '<20', '<40', '<180')
    y_pos = np.arange(len(bin_labels))

    fig = plt.figure()
    plt.bar(y_pos, bins.tolist(), align='center', alpha=0.5)
    plt.xticks(y_pos, bin_labels)
    for i in range(4):
        plt.text(y_pos[i], bins[i], str(bins[i]))
    return fig


def compute_histogram(model, dg, count_only=False, test_descriptors=None):
    if test_descriptors is None:
        test_descriptors = net.compute_descriptor(model, dg.test_loader)
    db_descriptors = net.compute_descriptor(model, dg.db_loader)

    angular_diffs = []
    for match in utils.knn_matching_search(test_descriptors, db_descriptors):
        m = dg.test_dataset.__getitem__(match.queryIdx)
        n = dg.db_dataset.__getitem__(match.trainIdx)
        if m['target'] == n['target']:
            angular_diffs.append(utils.compute_angle(m['pose'], n['pose']))

    bins = get_histogram_bins(angular_diffs)
    if count_only:
        return bins

    fig = get_histogram_plot(bins)
    return bins, fig


def plot_embedding(embedding, title, dg):
    colors = {'ape': 'green', 'benchvise': 'blue', 'cam': 'orange', 'cat': 'purple', 'duck': 'red'}
    fig, ax = plt.subplots()
    start_idx = 0
    test_labels_arr = np.array(dg.test_labels)
    classes = np.unique(dg.test_labels)
    for target in classes:
        end_idx = start_idx + len(test_labels_arr[test_labels_arr == target])
        sub_reps = embedding[start_idx:end_idx, :]
        ax.scatter(sub_reps[:, 0], sub_reps[:, 1], c=colors[target], label=target, alpha=0.3, edgecolors='none')
        start_idx = end_idx
    ax.legend()
    ax.set_title(title)
    return fig


def get_pca_plot(descriptors, dg):
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(descriptors)
    return plot_embedding(embedding, title="PCA Embedding of Test Database", dg=dg)


def get_tsne_plot(descriptors, dg):
    tsne = TSNE(n_components=2)
    embedding = tsne.fit_transform(descriptors)
    return plot_embedding(embedding, title='t-SNE Embedding of Coarse Database', dg=dg)


def compute_confusion_matrix(preds, trues):
    '''
    Computes a confusion matrix using numpy for two np.arrays true and pred.
    Results are identical (and similar in computation time) to "sklearn.metrics.confusion_matrix"
    '''
    num_class = len(np.unique(trues))  # Number of classes
    confusion_mat = np.zeros((num_class, num_class), dtype=np.uint32)

    for i in range(len(trues)):
        confusion_mat[trues[i]][preds[i]] += 1
    return confusion_mat


def get_confusion_heatmap(model, dg, test_descriptors=None):
    if test_descriptors is None:
        test_descriptors = net.compute_descriptor(model, dg.test_loader)
    db_descriptors = net.compute_descriptor(model, dg.db_loader)

    matches = utils.knn_matching_search(test_descriptors, db_descriptors)
    test_idx = []
    db_idx = []
    for m in matches:
        test_idx.append(m.queryIdx)
        db_idx.append(m.trainIdx)

    preds = np.array(dg.test_dataset.targets)[test_idx]
    trues = np.array(dg.db_dataset.targets)[db_idx]
    cm = compute_confusion_matrix(preds, trues)

    fig = plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')  # https://likegeeks.com/seaborn-heatmap-tutorial/
    plt.ylabel('True Label (DB)')
    plt.xlabel('Predicted Label (TEST)')
    plt.title('Confusion Matrix')
    return fig


def get_all_plots(model, dg, descriptors):
    _, hist_fig = compute_histogram(model, dg, test_descriptors=descriptors)
    pca_fig = get_pca_plot(descriptors.cpu().numpy(), dg)
    tsne_fig = get_tsne_plot(descriptors.cpu().numpy(), dg)
    confusion_fig = get_confusion_heatmap(model, dg, descriptors)
    return hist_fig, pca_fig, tsne_fig, confusion_fig
