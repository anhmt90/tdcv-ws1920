import matplotlib.pyplot as plt
import cv2

mean = [0.1173, 0.0984, 0.0915]
std = [0.2281, 0.1765, 0.1486]


def unnormalize(image):
    return image * std + mean


def visualize_triplet(anchor, puller, pusher):
    size = anchor['image'].size()[0]//4 # we are going to visualize only the first 4 images of the batch
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

    # Define the number of nearest neighbours
    knn = 1

    # create BFMatcher object. In the homework it is stated we should use Euclidean distance as the metric
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Match descriptors.
    match = bf.knnMatch(descriptors_testdataset, descriptors_dbdataset, k=knn)

    return match
