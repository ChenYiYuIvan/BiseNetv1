import torch.nn as nn
import torch
import torch.nn.functional as F

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)
        # intersect = (output * target).sum(-1).sum() + self.epsilon
        # denominator = ((output + target).sum(-1)).sum() + self.epsilon

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice = intersect / denominator
        dice = torch.mean(dice)
        return 1 - dice
        # return 1 - 2. * intersect / denominator

class HighEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-4
        self.eta = 2

    def forward(self, output):

        # copied from code of paper
        pred = F.softmax(output, dim=1)
        log_pred = F.log_softmax(output, dim=1)
        loss = pred * log_pred
        loss = -(loss.sum(dim=1)) # / 2.9444
        loss = self.charbonnier(loss)

        return loss.mean()

    def charbonnier(self, x):
        x = x**2 + self.eps**2
        x = x**self.eta
        return x
