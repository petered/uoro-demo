import inspect
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
from torch.autograd import Variable

from artemis.general.nested_structures import nested_map


def torch_str(var, thresh=10):

    if isinstance(var, (Variable, )):
        if np.prod(var.size())>thresh:
            return '<{} with size {}, min {}, max {}>'.format(var.__class__.__name__, var.size(), var.min().data[0], var.max().data[0])
        else:
            # return '<{} with values {}>'.format(var.__class__.__name__, var.data.numpy())
            return str(var.data.numpy())
    elif isinstance(var, (list, tuple, dict)):
        return nested_map(torch_str, var)
    else:
        raise Exception(var.__class__)


def torch_print(var, thresh=10):
    print (torch_str(var, thresh=thresh))


class TorchRandomState(object):

    def __init__(self, seed = None):
        # if seed is None:
        #     seed = torch.get_rng_state()
        # else:
        #
        old_seed = torch.get_rng_state()
        torch.manual_seed(seed)
        self.seed = torch.get_rng_state()
        torch.set_rng_state(old_seed)

    def _wrapped_call(self, f, size):
        old_state = torch.get_rng_state()
        torch.set_rng_state(self.seed)
        out = f(*size)
        self.seed = torch.get_rng_state()
        torch.set_rng_state(old_state)
        return out

    def randn(self, *size):
        return self._wrapped_call(torch.randn, size)

    def rand(self, *size):
        return self._wrapped_call(torch.rand, size)

#
# class CollectionVariable(Variable):
#
#     def __init__(self):


# def get_module_single_parameter():
#

def _torch_to_numpy(var):
    return var.data.numpy() if isinstance(var, torch.autograd.Variable) else var


def _numpy_to_torch_var(data, requires_grad=False, volatile=False, cast_floats = None):
    if cast_floats is not None and data.dtype in ('float32', 'float64'):
        data = data.astype(cast_floats)
    return torch.autograd.Variable(torch.from_numpy(data), requires_grad=requires_grad, volatile=volatile) if isinstance(data, np.ndarray) else data


def torch_struct_to_numpy_struct(obj):
    return nested_map(_torch_to_numpy, obj)


def numpy_struct_to_torch_struct(obj, cast_floats = None):
    return nested_map(partial(_numpy_to_torch_var, cast_floats=cast_floats), obj)


def numpify(torch_fcn):

    def numpified_function(*args, **kwargs):
        vargs = numpy_struct_to_torch_struct(args)
        vkwargs = numpy_struct_to_torch_struct(kwargs)
        result = torch_fcn(*vargs, **vkwargs)
        return torch_struct_to_numpy_struct(result)

    return numpified_function


def clone_em(vars):
    return nested_map(lambda x: x.clone() if isinstance(x, Variable) else x, vars)


@contextmanager
def magically_torchify_everything():
    """
    Yeah, this is very bad practice.  Very, very naughty.

    It also doesn't work.
    :return:
    """
    parent_namespace = inspect.currentframe().f_back.f_back.f_locals
    print(parent_namespace.keys())

    torchified_namespace = numpy_struct_to_torch_struct(parent_namespace)

    for k in parent_namespace.keys():
        parent_namespace[k] = torchified_namespace[k]

    yield

    numpified_namespace = torch_struct_to_numpy_struct(parent_namespace)

    for k in parent_namespace.keys():
        parent_namespace[k] = numpified_namespace[k]


def torch_loop(func, *args):

    n_steps = len(args[0])
    assert all(len(a) == n_steps for a in args[1:]), 'All arguments should have the same first dimension, which represents the number of steps.  Actual shapes are: {}'.format([a.size() for a in args])

    first_out = func(*(a[0] for a in args))
    if first_out is None:
        for i in range(1, n_steps):
            func(*(a[i] for a in args))
    else:
        # Should probably assert that it's a tensor
        out_tensor = torch.autograd.Variable(torch.zeros(n_steps, *first_out.size()))
        out_tensor[0] = first_out
        for i in range(1, n_steps):
            out_tensor[i] = func(*(a[i] for a in args))
        return out_tensor