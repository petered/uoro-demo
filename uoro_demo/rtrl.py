import torch
from torch.autograd import Variable, grad
from artemis.general.nested_structures import nested_map
from uoro_demo.torch_utils.interfaces import TrainableStatefulModule
from uoro_demo.torch_utils.torch_helpers import clone_em
from uoro_demo.torch_utils.training import get_named_torch_optimizer_factory, create_loss_function, \
    jacobian, set_optimizer_learning_rate
from uoro_demo.torch_utils.variable_workshop import make_single_module_parameter, MergedVariable


class RTRL(TrainableStatefulModule):
    """
    An Implementation of Real Time Recurrent Learning
    """

    def __init__(self, forward_update_module, loss = 'mse', optimizer_factory = get_named_torch_optimizer_factory('sgd', 0.01),
                 learning_rate_generator = None):
        """
        :param forward_update_module: a RecurrentStatelessModule whose forward function has the form
            output, state = forward_update_module(input, state)
        :param loss: A string identifying a loss (see create_loss_function) or a loss function of the form
            loss = f_loss(prediction, target)
        :param optimizer_factory: Construct with get_named_torch_optimizer_factory
        :param learning_rate_generator: None to keep a fixed learning rate, or a generator that returns a new learning
            rate at each iteration.
        """

        super(RTRL, self).__init__()
        self.forward_update_module = forward_update_module
        self.state = None
        self.theta = make_single_module_parameter(forward_update_module)
        self.loss_function = loss if callable(loss) else create_loss_function(loss)
        self.optimizer = optimizer_factory([self.theta])
        self.dstate_dtheta = None
        self.learning_rate_generator = learning_rate_generator

    def _initialize_state(self, x):
        self.state = self.forward_update_module.get_initial_state(x)

    def forward(self, x):
        if self.state is None:
            self._initialize_state(x)
        out, self.state = self.forward_update_module(x, nested_map(lambda x: x.detach(), self.state))
        return out

    def train_it(self, x, y):
        if self.state is None:
            self._initialize_state(x)
        state_vec_old = MergedVariable.join(self.state, requires_grad=True, as_leaf=True)
        if self.dstate_dtheta is None:
            self.dstate_dtheta = Variable(torch.zeros(len(state_vec_old), len(self.theta)))
        out, state_new = self.forward_update_module(x, state_vec_old.cleave())
        loss = self.loss_function(out, y)

        dl_dstate_old, dl_dtheta_direct = grad(loss, (state_vec_old, self.theta), retain_graph=True)
        state_vec_new = MergedVariable.join(state_new)
        d_state_new_d_state_old = jacobian(state_vec_new, state_vec_old)
        d_state_new_d_theta_direct = jacobian(state_vec_new, self.theta)

        # Updates:
        self.theta.grad = dl_dstate_old.view(1, -1).mm(self.dstate_dtheta).view(-1) + dl_dtheta_direct
        self.dstate_dtheta = d_state_new_d_state_old.mm(self.dstate_dtheta) + d_state_new_d_theta_direct

        self.state = state_new
        if self.learning_rate_generator is not None:
            set_optimizer_learning_rate(self.optimizer, next(self.learning_rate_generator))
        self.optimizer.step()

        return out

    def get_state(self):
        return clone_em((self.theta, self.state, self.dstate_dtheta))

    def set_state(self, state):
        self.theta, self.state, self.dstate_dtheta = state
