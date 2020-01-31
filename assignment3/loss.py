import torch


def triplet_loss(output):
    batch_size = output.size()[0]
    diff_pos = output[0:batch_size:3] - output[1:batch_size:3]
    diff_neg = output[0:batch_size:3] - output[2:batch_size:3]
    norm_pos = torch.norm(diff_pos.view(batch_size//3, -1), p=2, dim=1)
    norm_neg = torch.norm(diff_neg.view(batch_size//3, -1), p=2, dim=1)
    loss = torch.max(torch.zeros(batch_size//3).to(output.device), 1 - (norm_pos**2 / (norm_neg**2 + 0.01)))
    return loss.sum()


def pairs_loss(output):
    batch_size = output.size()[0]
    diff_pos = output[0:batch_size:3] - output[1:batch_size:3]
    loss = torch.norm(diff_pos.view(batch_size//3, -1), p=2, dim=1)
    loss = loss**2
    return loss.sum()


def total_loss(output):
    batch_size = output.size()[0]
    return (triplet_loss(output) + pairs_loss(output)) / (batch_size//3)

