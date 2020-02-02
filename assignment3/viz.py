import matplotlib.pyplot as plt
import numpy as np
import torch


import utils
import net
import data_generator as dgr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def visualize_triplet(anchor, puller, pusher, num_display=4):
    fig, ax = plt.subplots(num_display, 3, figsize=(50, 50))

    for i in range(num_display):
        ax[i, 0].imshow(dgr.train_denormalize(anchor['image'][i].numpy().transpose(1, 2, 0)))
        ax[i, 1].imshow(dgr.db_denormalize(puller['image'][i].numpy().transpose(1, 2, 0)))
        ax[i, 2].imshow(dgr.db_denormalize(pusher['image'][i].numpy().transpose(1, 2, 0)))
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        ax[i, 2].set_axis_off()
    fig.show()


def make_histogram(data):
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


def visualize_histogram(angular_diffs):
    bins = make_histogram(angular_diffs)

    bin_labels = ('<10', '<20', '<40', '<180')
    y_pos = np.arange(len(bin_labels))
    plt.bar(y_pos, bins.tolist(), align='center', alpha=0.5)
    plt.xticks(y_pos, bin_labels)
    for i in range(4):
        plt.text(y_pos[i], bins[i], str(bins[i]))
    plt.show()


def store_embeddings(writer, embeddings, metadata, label_img=False):
    '''
    :param writer: instance of SummaryWriter()
    :param embeddings: the tesnsor to visualized by PCA and t-SNE
    :param metadata: labels of the embeddings e.g. ape, benchvise, cat, cam, duck
    :param label_img: if True, the RGB images will be plotted in place of points i.e. cloud images instead of cloud points
    :return:
    '''
    assert torch.is_tensor(embeddings), "'descriptors' is not a Tensor object"
    test_images = torch.cat([test_input['image'] for j, test_input in enumerate(dg.test_loader)]) if label_img else None

    writer.add_embedding(embeddings, metadata=metadata, label_img=test_images, tag='test_descriptors')


def compute_histogram(model, dg, logging=False, test_descriptors=None):
    if test_descriptors is None:
        test_descriptors = torch.cat([model(test_input['image']) for j, test_input in enumerate(dg.test_loader)])
    db_descriptors = torch.cat([model(db_input['image']) for j, db_input in enumerate(dg.db_loader)])

    if str(device) == 'cpu':
        test_descriptors = test_descriptors.cpu().numpy()
        db_descriptors = db_descriptors.cpu().numpy()

    true_positives = 0
    angular_diffs = []
    for match in utils.knn_to_dbdataset(test_descriptors, db_descriptors):
        m = dg.test_dataset.__getitem__(match.queryIdx)
        n = dg.db_dataset.__getitem__(match.trainIdx)
        if m['target'] == n['target']:
            if (logging):
                angular_diffs.append(utils.compute_angle(m['pose'], n['pose']))
            else:
                true_positives += 1

    visualize_histogram(angular_diffs)
    return angular_diffs if not logging else true_positives
