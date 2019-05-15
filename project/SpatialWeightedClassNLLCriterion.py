from torch.nn.modules import Module
from torch._jit_internal import weak_module, weak_script
from torch.autograd.variable import Variable

import torch.nn.functional as F
import torch.nn._reduction as _Reduction


def _make_function_weighted_class_criterion(class_name, update_output, update_grad_input, acc_grad_parameters):
    weight_arg_idx = -1

    for i, arg in enumerate(update_output.arguments):
        if arg.name.startswith('weight') and arg.name != 'weight_map':
            weight_arg_idx = i
            break

    buffers_idx = []
    additional_arg_idx = 0
    for arg in update_output.arguments[5:]:
        if not arg.name.startswith('weight') and arg.type == 'THTensor*':
            buffers_idx.append(additional_arg_idx)
        additional_arg_idx += 1

    def __init__(self, *args, **kwargs):
        Function.__init__(self)
        self.weight = kwargs.get('weight')
        self.additional_args = list(args)

    def forward(self, input, target, weight_map):
        self._backend = type2backend[type(input)]
        self.save_for_backward(input, target, weight_map)
        if weight_arg_idx >= 0:
            insert_idx = weight_arg_idx - 5  # state, input, target, weight_map, output
            self.additional_args.insert(insert_idx, self.weight)
        for idx in buffers_idx:
            self.additional_args.insert(idx, input.new(1))
        output = input.new(1)
        getattr(self._backend, update_output.name)(self._backend.library_state, input, target, weight_map,
                                                   output, *self.additional_args)
        return output

    def backward(self, grad_output):
        input, target, weight_map = self.saved_tensors
        grad_input = grad_output.new().resize_as_(input).zero_()
        getattr(self._backend, update_grad_input.name)(self._backend.library_state, input, target, weight_map,
                                                       grad_input, *self.additional_args)
        grad_output_expanded = grad_output.view(*repeat(1, grad_input.dim()))
        grad_input.mul_(grad_output_expanded.expand_as(grad_input))
        return grad_input, None

    return type(class_name, (Function,), dict(__init__=__init__, forward=forward, backward=backward))


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


class SpatialWeightedClassNLLCriterion(_WeightedLoss):
    def __init__(self, *args, **kwargs):
        Function.__init__(self)
        self.weight = kwargs.get('weight')
        self.additional_args = list(args)

    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(SpatialWeightedClassNLLCriterion, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index