import matplotlib.pyplot as plt


mean = [0.1173, 0.0984, 0.0915]
std = [0.2281, 0.1765, 0.1486]


def unnormalize(image):
    return image * std + mean


def visualize_triplet(anchor, puller, pusher):
    size = anchor['image'].size()[0]//4
    fig, ax = plt.subplots(size, 3, figsize=(50,50))

    for i in range(size):
        ax[i, 0].imshow(unnormalize(anchor['image'][i].numpy().transpose(1, 2, 0)))
        ax[i, 1].imshow(unnormalize(puller['image'][i].numpy().transpose(1, 2, 0)))
        ax[i, 2].imshow(unnormalize(pusher['image'][i].numpy().transpose(1, 2, 0)))
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        ax[i, 2].set_axis_off()
    fig.show()
