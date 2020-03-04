import os
import torch
from datetime import datetime

import net
import viz
import data_generator
import datasets
from torch.utils.tensorboard import SummaryWriter

'''
    How to run tensorboard: https://www.endtoend.ai/blog/pytorch-tensorboard/
'''


def run(dg):
    print("Start creating logs for all checkpoints")
    writer = SummaryWriter()
    checkpoint_files = datasets.sorted_alphanumeric(os.listdir('models'))

    start_time = datetime.now()
    for i, file in enumerate(checkpoint_files):
        _, ext = os.path.splitext(file)
        if ext.lower() != '.pt' or file == 'best.pt':
            continue

        print(file)
        ckp_path = os.path.join('models', file)
        model = net.Net()
        model, _ = net.load_ckp(ckp_path, model)
        test_descriptors = net.compute_descriptor(model, dg.test_loader)

        _, hist_fig = viz.compute_histogram(model, dg, test_descriptors=test_descriptors)
        confusion_fig = viz.get_confusion_heatmap(model, dg, test_descriptors)
        writer.add_figure("Histogram", hist_fig, global_step=i)
        writer.add_figure("Confusion Heatmap", confusion_fig, global_step=i)
        writer.add_embedding(test_descriptors, metadata=dg.test_labels, tag='test_descriptors')
        # writer.add_embedding(test_descriptors, metadata=dg.test_labels, label_img=dg.test_images,
        #  tag='test_descriptors (w/ images)')

        # pca_fig = viz.get_pca_plot(test_descriptors.cpu().numpy(), dg)
        # tsne_fig = viz.get_tsne_plot(test_descriptors.cpu().numpy(), dg)
        # writer.add_figure("PCA", pca_fig, global_step=i)
        # writer.add_figure("t-SNE", tsne_fig, global_step=i)

    timeElapsed = datetime.now() - start_time
    print('Finished! Time elapsed (hh:mm:ss.ms) {}'.format(timeElapsed))
    writer.close()

if __name__ == '__main__':
    dg = data_generator.DataGenerator(root='./dataset', batch_size=4096)
    run(dg)





