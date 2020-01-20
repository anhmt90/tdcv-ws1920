import torch

def triplet_loss(output):
    batch_size = output.size()[0]
    diff_pos = output[0:batch_size:3] - output[1:batch_size:3]
    diff_neg = output[0:batch_size:3] - output[2:batch_size:3]
    loss = torch.max(torch.zeros(batch_size), -1 * (torch.sqrt(diff_neg).pow(2).sum(1) / (torch.sqrt(diff_pos).pow(2).sum(1) + 0.01)) + 1)
    loss = torch.sum(loss)
    return loss  


def pairs_loss(output):
    batch_size = output.size()[0]
    diff_pos = output[0:batch_size:3] - output[1:batch_size:3]
    loss = torch.sqrt(diff_pos).pow(2).sum(1)
    loss = torch.sum(loss)
    return loss

def total_loss(output):
    batch_size = output.size()[0]
    return (triplet_loss(output) + pairs_loss(output)) / batch_size

