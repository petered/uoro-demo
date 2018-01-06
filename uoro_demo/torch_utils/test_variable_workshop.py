import numpy as np
import torch
from torch.autograd import Variable, grad

from uoro_demo.torch_utils.variable_workshop import TorchVariableBuffer, JoinCleave, MergedVariable, \
    make_single_module_parameter


def test_variable_buffer():

    buff = TorchVariableBuffer(length=3)

    d = buff(torch.Tensor([0, 1]))
    assert np.array_equal(d.data.numpy(), [[0, 1]])

    d = buff(torch.Tensor([3, 4]))
    assert np.array_equal(d.data.numpy(), [[0, 1], [3, 4]])

    d = buff(torch.Tensor([5, 6]))
    assert np.array_equal(d.data.numpy(), [[0, 1], [3, 4], [5, 6]])

    d = buff(torch.Tensor([7, 8]))
    assert np.array_equal(d.data.numpy(), [[3, 4], [5, 6], [7, 8]])

    d = buff(torch.Tensor([9, 10]))
    assert np.array_equal(d.data.numpy(), [[5, 6], [7, 8], [9, 10]])


def test_join_split():
    x1 = Variable(torch.randn(5, 4))
    x2 = Variable(torch.randn(3, ))

    js = JoinCleave()
    vec = js.join((x1, x2))
    assert vec.size() == (5*4+3, )

    x1_new, x2_new = js.cleave(vec)

    assert torch.equal(x1.data, x1_new.data)
    assert torch.equal(x2.data, x2_new.data)


def test_join_split_single():
    x1 = Variable(torch.randn(5, 4))
    js = JoinCleave()
    vec = js.join(x1)
    assert vec.size() == (5*4, )
    x1_new = js.cleave(vec)

    assert torch.equal(x1.data, x1_new.data)


def test_join_split_dim():

    x1 = Variable(torch.randn(5, 4))
    x2 = Variable(torch.randn(5, 3, 2))

    js = JoinCleave(dim=1)
    vec = js.join((x1, x2))

    assert vec.size() == (5, 4+3*2)

    x1_new, x2_new = js.cleave(vec)

    assert torch.equal(x1.data, x1_new.data)
    assert torch.equal(x2.data, x2_new.data)

    assert torch


def test_merged_variable_grads():

    x = Variable(torch.randn(5, 4))
    w1 = Variable(torch.randn(4, 3), requires_grad=True)
    w2 = Variable(torch.randn(3, 2), requires_grad=True)
    hloss = ((x.mm(w1))**2).sum()

    dhloss_dw1, = grad(hloss, w1, retain_graph=True)

    theta = MergedVariable.join([w1, w2], requires_grad=True, as_leaf=True)

    w1_, w2_ = theta.cleave()
    hloss_ = ((x.mm(w1_))**2).sum()
    dhloss_dw1_, = grad(hloss_, w1_, retain_graph=True)
    assert torch.equal(dhloss_dw1.data, dhloss_dw1_.data)

    dhloss_dtheta, = grad(hloss_, theta)
    dhloss_dw1_2, dhloss_dw2_2 = theta.cleave(dhloss_dtheta)
    assert torch.equal(dhloss_dw1_2.data, dhloss_dw1_.data)
    assert torch.equal(dhloss_dw2_2.data, torch.zeros(*dhloss_dw2_2.size()))


def test_make_single_module_parameter():

    module = torch.nn.Linear(5, 6)
    assert len(list(module.parameters()))==2
    param = make_single_module_parameter(module)
    old_param_value = param.data.clone()
    # assert len(list(module.parameters()))==0
    assert param.size() == (5*6+6, )
    optim = torch.optim.SGD(params=[param], lr=0.2)
    x = Variable(torch.randn(3, 5))
    loss = (module(x)**2).sum()
    loss.backward()
    optim.step()
    new_param_value = param.data.clone()
    assert not torch.equal(old_param_value, new_param_value), 'Parameter has not changed'

    loss2 = (module(x)**2).sum()
    assert not torch.equal(loss.data, loss2.data), 'Parameter change has not affected module function'

    loss3 = (module(x)**2).sum()
    assert torch.equal(loss2.data, loss3.data), 'Module is is not behaving consistently between calls'

    param.data[4] += 1
    loss4 = (module(x)**2).sum()
    assert not torch.equal(loss3.data, loss4.data), 'Manually changing parameter did not affect module function'


def test_single_module_parameter_converges():
    x = Variable(torch.randn(3, 5))
    y = Variable(torch.randn(3, 2))

    module = torch.nn.Linear(5, 6)








if __name__ == '__main__':
    # test_variable_buffer()
    # test_join_split()
    # test_join_split_single()
    # test_join_split_dim()
    # test_merged_variable_grads()
    test_make_single_module_parameter()
