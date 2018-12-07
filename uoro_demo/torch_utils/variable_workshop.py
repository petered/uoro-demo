import numpy as np
import torch
from torch.nn import Parameter

from artemis.general.nested_structures import NestedType, nested_map
from artemis.general.should_be_builtins import izip_equal


class TorchVariableBuffer(object):

    def __init__(self, length):
        self.length = length
        self.buffer = None
        self.t = 0

    # def __call__(self, x):
    #     if self.buffer is None:
    #         self.buffer = torch.autograd.Variable(torch.zeros(self.length, *x.size()))
    #     else:
    #         self.buffer[:-1] = self.buffer[1:].clone()
    #     self.buffer[-1] = x
    #     self.t+=1
    #     return self.buffer[max(0, self.length-self.t):]

    def __call__(self, x):
        if self.buffer is None:
            self.buffer = torch.autograd.Variable(torch.zeros(self.length, *x.size()))

        self.buffer[self.t % self.length] = x
        self.t+=1

        ixs = (torch.arange(max(0, self.t-self.length), self.t) % self.length).long()

        return self.buffer[ixs]

        # else:
        #     self.buffer[:-1] = self.buffer[1:].clone()
        #
        #
        #
        # self.buffer[-1] = x
        # self.t+=1
        # return self.buffer[max(0, self.length-self.t):]


class JoinCleave(object):

    def __init__(self, dim = 0):
        """
        :param dim: Dimension after which to flatten and join all tensors.  The resulting tensor will be (dim+1) dimensional.
        """
        self.dim = dim
        self._split_axis_sizes = None

    def join(self, x):
        if self._split_axis_sizes is None:
            self._nested_type = NestedType.from_data(x)
            data_list = self._nested_type.get_leaves(x)
            self._split_shapes = [x_.size() for x_ in data_list]
            self._pre_join_shapes = [list(x_.size()[:self.dim])+[np.prod(list(x_.size()[self.dim:]))] for x_ in data_list]
            self._split_axis_ixs = torch.cumsum(torch.IntTensor([0] + [s_[-1] for s_ in self._pre_join_shapes]), dim=0)
        else:
            data_list = self._nested_type.get_leaves(x)

        return torch.cat(list(x_.view(*s_) for x_, s_ in izip_equal(data_list, self._pre_join_shapes)), dim=self.dim)

    def cleave(self, x_flat, share_data = True):
        """
        Split (or cleave) a variable into its component parts..
        :param x_flat:
        :param share_data:
        :return:
        """
        if share_data:
            x_split = [x_flat[..., start:end].view(shape) for (start, end, shape) in izip_equal(self._split_axis_ixs[:-1], self._split_axis_ixs[1:], self._split_shapes)]
        else:  # Note: this will raise an Error if the self.dim != 0, because the data is no longer contigious in memory.
            x_split = [x_flat[..., start:end].clone().view(shape) for (start, end, shape) in izip_equal(self._split_axis_ixs[:-1], self._split_axis_ixs[1:], self._split_shapes)]
        x_reassembled = self._nested_type.expand_from_leaves(x_split, check_types=False)
        return x_reassembled


class MergedVariable(torch.Tensor):

    @staticmethod
    def join(vars, dim=0, requires_grad=False, as_leaf = False):
        join_split = JoinCleave(dim=dim)

        merged_var = join_split.join(vars)

        if as_leaf:
            merged_var = torch.tensor(merged_var.data, requires_grad=requires_grad)

        merged_var.__class__ = MergedVariable
        merged_var._join_split = join_split
        return merged_var

    def cleave(self, other = None, share_data=True, to_params = False):
        if other is not None:
            assert other.size() == self.size()
        else:
            other = self
        split_vars = self._join_split.cleave(other, share_data=share_data)
        if to_params:
            split_vars = nested_map(lambda v: Parameter(v.data), split_vars)
        return split_vars


def set_module_parameters(module, parameters_iterator, _assert_done = True, only_requiring_grad=True):
    """
    Replace the parameters of the module with parameters pulled from the parameters_iterator.



    :param module:
    :param parameters_iterator:
    :param _assert_done:
    :return:
    """
    # See http://pytorch.org/docs/master/_modules/torch/nn/modules/module.html#Module.named_parameters
    # First set locals, then dig.

    if isinstance(parameters_iterator, (list, tuple)):
        parameters_iterator = (p for p in parameters_iterator)

    for name, p in module._parameters.items():
        if p is not None and ((not only_requiring_grad) or p.requires_grad):
            # Sometimes a param can be None, as when you disable the bias.
        # setattr(module, name, next(parameters_iterator))
            module.__dict__[name] = next(parameters_iterator)
        # del module._parameters[name]

    for name, child in module.named_children():
        set_module_parameters(child, parameters_iterator, _assert_done=False)

    if _assert_done:
        try:
            next(parameters_iterator)
        except StopIteration:
            pass
        except:
            raise Exception('Parameters were not all used!')


def make_single_module_parameter(module, only_requiring_grad = True):
    """
    This is a nice function.
    :param module:
    :param only_requiring_grad:
    :return:
    """
    params = list(p for p in module.parameters() if (not only_requiring_grad) or p.requires_grad)
    single_param = MergedVariable.join(params, requires_grad=True, as_leaf=True)
    new_params = single_param.cleave(share_data = True)
    set_module_parameters(module, (p for p in new_params), only_requiring_grad = only_requiring_grad)
    # assert len(list(module.parameters()))==0
    return single_param
