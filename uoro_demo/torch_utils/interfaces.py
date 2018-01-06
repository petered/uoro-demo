from abc import ABCMeta, abstractmethod

import torch

from uoro_demo.torch_utils.torch_helpers import torch_loop
from uoro_demo.torch_utils.training import create_loss_function


class TorchGradientBasedPredictor(torch.nn.Module):

    def __init__(self, forward_update_module, loss, optimizer_factory, retain_graph = False):
        super(TorchGradientBasedPredictor, self).__init__()
        self.predictor = forward_update_module
        self.loss = create_loss_function(loss)
        self.optimizer = optimizer_factory(forward_update_module.parameters())
        self.retain_graph = retain_graph

    def forward(self, *xs):
        self.predictor.train(False)
        out = self.predictor(*xs)
        self.predictor.train(True)
        return out

    def train_it(self, x, y):
        self.predictor.train(True)
        prediction = self.predictor(x)
        self.zero_grad()
        loss = self.loss(prediction, y)
        loss.backward(retain_graph=self.retain_graph)
        self.optimizer.step()
        self.predictor.train(False)


class ITrainableModule(torch.nn.Module):

    __metaclass__ = ABCMeta

    @abstractmethod
    def train_it(self, x, y):
        """
        Perform a single iteration of training
        :param x: An input variable
        :param y: A target variable
        :return: The prediction for the input
        """
        raise NotImplementedError()


class StatefulModule(torch.nn.Module):

    def get_state(self):
        return None

    def set_state(self, state):
        assert state is None


class StatefulGradientBasedPredictor(TorchGradientBasedPredictor, StatefulModule):

    def get_state(self):
        return self.predictor.get_state()

    def set_state(self, state):
        self.predictor.set_state(state)


class WrapAsStateful(StatefulModule):

    def __init__(self, forward_module):
        super(WrapAsStateful, self).__init__()
        self.forward_module = forward_module

    def forward(self, *x):
        return self.forward_module(*x)


class TrainableStatefulModule(ITrainableModule, StatefulModule):

    pass


class LoopingPredictor(TrainableStatefulModule):
    """
    Just applies a predictor in a loop
    """

    def __init__(self, predictor):
        super(LoopingPredictor, self).__init__()
        self.predictor = predictor

    def forward(self, x):
        """
        :param x: n_samples, batch_size, n_dim
        :return:
        """
        y = torch_loop(self.predictor, x)
        return y

    def train_it(self, x, y):
        """
        Perform a single iteration of training
        :param x: An input variable
        :param y: A target variable
        :return: The prediction for the input
        """
        return torch_loop(self.predictor.train_it, x, y)


class RecurrentStatelessModule(torch.nn.Module):
    """
    A function of the form:

        y, s_new = f(x, s_old)

    """

    @abstractmethod
    def get_initial_state(self, x_init):
        raise NotImplementedError()

    @abstractmethod
    def forward(self, x, state):
        """
        :param x: A (batch_size, dim_x) array of inputs
        :param state: A (batch_size, dim_s) array of state variables
        :return: y, new_state
            y: A (batch_size, dim_y) array of outputs
            state: A (batch_size, dim_s) array of updated state variables.
        """
        raise NotImplementedError()