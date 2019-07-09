from torch._jit_internal import weak_module, weak_script

import torch
import torch.nn.modules.loss as nn_loss


"""
This file contains modified source-code from PyTorch that 
was adjusted to also work with our weight maps.

The parameters weight should not be confused with the
weight maps as these are different.
"""

@weak_module
class WeightedCrossEntropyLoss(nn_loss._WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, device, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.device = device
        self.ignore_index = ignore_index


    def forward(self, input, target, wmap=None):
        """
        The method that is automatically called when the
        prediction, label and possible weight map are given
        to the instantiated object of this class.
        """

        return self.weighted_cross_entropy(input, target, wmap)

    @weak_script
    def weighted_cross_entropy(self, output, target, wmap=None):
        """
        Computes the individual cross-entropy loss values for
        each class and averages these. It also takes in account
        the weight maps when applicable.
        """

        p = self.softmax(output)

        negatives, positives = self.check_mistakes(output, target)

        loss = 0
        classes = len(output[1, :])
        for layer in range(classes):
            if wmap is not None:
                mistakes = negatives * p[:, layer]
                correct = positives * (1 - p[:, layer])

                loss += -(torch.log(correct + mistakes) * wmap).mean()

            else:
                mistakes = negatives * p[:, layer]
                correct = positives * (1 - p[:, layer])

                loss += -torch.log(correct + mistakes).mean()

        return loss / classes


    def softmax(self, output):
        """
        Computes the probability that a pixel was
        classified correct for each pixel.
        """

        return torch.exp(output) / torch.exp(output).sum()


    def check_mistakes(self, output, target):
        """
        Returns the binary indicators that specify
        which pixels were correctly classified or not.
        """

        output = torch.argmax(output, 1)
        target = torch.argmax(target, 1)

        mistakes = torch.ne(output, target).type(torch.FloatTensor).to(device=self.device)
        correct = torch.eq(output, target).type(torch.FloatTensor).to(device=self.device)

        return [mistakes, correct]
