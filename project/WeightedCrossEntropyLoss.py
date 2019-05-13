from torch.nn.modules import Module
from torch._jit_internal import weak_module, weak_script

import torch
import torch.nn.functional as F
import torch.nn._reduction as _Reduction
import warnings

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
    r"""This criterion combines `log_softmax` and `nll_loss` in a single
    function.

    See :class:`~torch.nn.CrossEntropyLoss` for details.

    Args:
        input (Tensor) : :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss.
        target (Tensor) : :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
            K-dimensional loss.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    """

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return nll_loss(input, target, wmap, weight, None, ignore_index, None, reduction)


@weak_script
def nll_loss(input, target, wmap, weight=None, size_average=None, ignore_index=-100,
             reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor
    r"""The negative log likelihood loss.

    See :class:`~torch.nn.NLLLoss` for details.

    Args:
        input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss.
        target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
            K-dimensional loss.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    """

    # if size_average is not None or reduce is not None:
    #     reduction = _Reduction.legacy_get_string(size_average, reduce)

    dim = input.dim()

    if dim != 2:
        raise ValueError('Expected 2 dimensions (got {})'.format(dim))

    if input.size(0) != target.size(0) or input.size(0) != wmap.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}) and wmap batch_size ({}).'
                         .format(input.size(0), target.size(0), wmap.size(0)))

    # ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)     # TODO: Continue from here.

    H = 164
    W = 164
    batch_size = 1

    # Calculate log probabilities
    logp = F.log_softmax(input)

    # Gather log probabilities with respect to target
    logp = logp.gather(1, target.view(batch_size, 1, H, W))

    # Multiply with weights
    weighted_logp = (logp * wmap).view(batch_size, -1)

    # Rescale so that loss is in approx. same interval
    weighted_loss = weighted_logp.sum(1) / wmap.view(batch_size, -1).sum(1)

    # Average over mini-batch
    weighted_loss = weighted_loss.mean()

    return weighted_loss
