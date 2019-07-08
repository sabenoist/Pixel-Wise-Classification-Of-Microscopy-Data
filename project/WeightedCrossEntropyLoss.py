from torch.nn.modules import Module
from torch._jit_internal import weak_module, weak_script
from torch.autograd.variable import Variable
import torch
import gc
import time

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

    def __init__(self, device, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.device = device
        self.ignore_index = ignore_index


    def forward(self, input, target, wmap=None):
        return self.weighted_cross_entropy(input, target, wmap)

    @weak_script
    def weighted_cross_entropy(self, output, target, wmap=None):

        # pixels_amount = output.shape[0]

        # logp = torch.log(self.softmax(output))
        # p = F.softmax(input)
        p = self.softmax(output)

        # print(logp)

        negatives, positives = self.check_mistakes(output, target)

        loss = 0
        classes = len(output[1,:])
        for layer in range(classes):
            if wmap is not None:
                mistakes = negatives * p[:,layer]
                correct = positives * (1 - p[:,layer])

                loss += -torch.log(correct + mistakes).mean()

            else:
                mistakes = negatives * p[:, layer]
                correct = positives * (1 - p[:, layer])

                loss += -torch.log(correct + mistakes).mean()


        return loss / classes


    def softmax(self, output):
        return torch.exp(output) / torch.exp(output).sum()


    def check_mistakes(self, output, target):
        output = torch.argmax(output, 1)
        target = torch.argmax(target, 1)

        mistakes = torch.ne(output, target).type(torch.FloatTensor).to(device=self.device)
        correct = torch.eq(output, target).type(torch.FloatTensor).to(device=self.device)

        return [mistakes, correct]


    def temp(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
            except:
                pass
