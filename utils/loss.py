import numpy as np
import torch
import os


def macro_soft_f1(output, label):
    output = output.reshape((output.shape[0], -1))
    label = label.reshape((label.shape[0], -1))
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """

    tp = torch.sum(output * label, dim=0)
    fp = torch.sum(output * (1 - label), dim=0)
    fn = torch.sum((1 - output) * label, dim=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = torch.mean(cost) # average on all labels
    return macro_cost