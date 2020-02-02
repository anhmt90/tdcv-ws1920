import os
import torch
import net
import viz
import data_generator
from torch.utils.tensorboard import SummaryWriter

'''
    How to use tensorboard: https://www.endtoend.ai/blog/pytorch-tensorboard/
'''


if __name__ == '__main__':
    dg = data_generator.DataGenerator(root='./dataset')
    writer = SummaryWriter()

    ckp_path = os.path.join('./models', 'checkpoint6.pt')
    model = net.Net()
    model, _ = net.load_ckp(ckp_path, model)

    test_descriptors = torch.cat([model(test_input['image']) for j, test_input in enumerate(dg.test_loader)])
    viz.compute_histogram(model, dg, test_descriptors=test_descriptors)
    viz.store_embeddings(writer, test_descriptors, dg.test_labels)
