__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["l2_loss",
           "square_loss", "sigmoid_cross_entropy", "pointwise_loss",
           "bpr_loss", "hinge", "pairwise_loss"]

import torch
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict



def l2_loss(*weights):
    """L2 loss

    Compute  the L2 norm of tensors without the `sqrt`:

        output = sum([sum(w ** 2) / 2 for w in weights])

    Args:
        *weights: Variable length weight list.

    """
    loss = 0.0
    for w in weights:
        loss += torch.sum(torch.pow(w, 2))

    return 0.5*loss
