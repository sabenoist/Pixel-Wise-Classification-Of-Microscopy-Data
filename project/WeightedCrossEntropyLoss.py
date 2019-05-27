from torch.nn.modules import Module
from torch._jit_internal import weak_module, weak_script
from torch.autograd.variable import Variable

import torch.nn.functional as F
import torch.nn._reduction as _Reduction

'''This file contains modified source-code from PyTorch that was adjusted to also work with our weight maps'''

class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)


@weak_module
class WeightedCrossEntropyLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index


    def forward(self, input, target, wmap):
        return weighted_cross_entropy(input, target, wmap, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)


def weighted_cross_entropy(input, target, wmap, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return nll_loss(input, target, wmap, weight, None, ignore_index, None, reduction)


@weak_script
def nll_loss(input, target, wmap, weight=None, size_average=None, ignore_index=-100,
             reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor

    dim = input.dim()

    if dim != 2:
        raise ValueError('Expected 2 dimensions (got {})'.format(dim))

    if input.size(0) != target.size(0) or input.size(0) != wmap.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}) and wmap batch_size ({}).'
                         .format(input.size(0), target.size(0), wmap.size(0)))

    H = 164
    W = 164
    batch_size = 1

    wmap = Variable(wmap)

    # Calculate log probabilities
    logp = F.log_softmax(input)

    # Gather log probabilities with respect to target
    logp = logp.gather(1, target.view(H * W, 5))    #  = softmaxed input - labels

    # Multiply with weights
    weighted_logp = (logp * wmap).view(batch_size, -1)

    # Rescale so that loss is in approx. same interval
    # weighted_loss = weighted_logp.sum(1) / wmap.view(batch_size, -1).sum(1)
    weighted_loss = weighted_logp / (H * W)

    # Average over mini-batch
    weighted_loss = -weighted_loss.mean()

    return weighted_loss
