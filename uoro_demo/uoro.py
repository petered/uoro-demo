import torch
from torch.autograd import grad, Variable
from artemis.general.nested_structures import nested_map
from uoro_demo.torch_utils.interfaces import TrainableStatefulModule
from uoro_demo.torch_utils.torch_helpers import clone_em
from uoro_demo.torch_utils.training import create_loss_function, get_named_torch_optimizer_factory, \
    set_optimizer_learning_rate
from uoro_demo.torch_utils.variable_workshop import MergedVariable, make_single_module_parameter


class UOROVec(TrainableStatefulModule):
    """
    An implementation of Unbiased Online Recurrent Optimization
    Corentin Tallec, Yann Ollivier
    https://arxiv.org/abs/1702.05043
    """

    def __init__(self, forward_update_module, loss = 'mse', optimizer_factory = get_named_torch_optimizer_factory('sgd', 0.01),
                 epsilon_perturbation = 1e-7, epsilon_stability = 1e-7, learning_rate_generator = None,
                 ):
        """
        :param forward_update_module: a RecurrentStatelessModule whose forward function has the form
            output, state = forward_update_module(input, state)
        :param loss: A string identifying a loss (see create_loss_function) or a loss function of the form
            loss = f_loss(prediction, target)
        :param optimizer_factory: Construct with get_named_torch_optimizer_factory
        :param epsilon_perturbation: The epsilon to use for tangent propagation
        :param epsilon_stability: The epsilon to use for stability in calculating rhos
        :param learning_rate_generator: None to keep a fixed learning rate, or a generator that returns a new learning
            rate at each iteration.
        """
        super(UOROVec, self).__init__()
        self.forward_update_module = forward_update_module
        self.theta = make_single_module_parameter(forward_update_module)
        self._state = None
        self.loss_function = loss if callable(loss) else create_loss_function(loss)
        self.s_toupee = None
        self.theta_toupee = None
        self.epsilon_perturbation = epsilon_perturbation
        self.epsilon_stability = epsilon_stability
        self.optimizer = optimizer_factory([self.theta])
        self.nu_gen = None
        self.learning_rate_generator = learning_rate_generator

    def _initialize_state(self, x):
        self._state = self.forward_update_module.get_initial_state(x)

    def forward(self, x):

        if self._state is None:
            self._initialize_state(x)

        out, self._state = self.forward_update_module(x, nested_map(lambda x: x.detach(), self._state))
        return out

    def train_it(self, x, y):

        if self._state is None:
            self._initialize_state(x)
        state_vec_old = MergedVariable.join(self._state, requires_grad=True, as_leaf=True)
        out, state_new = self.forward_update_module(x, state_vec_old.cleave())
        state_vec_new = MergedVariable.join(state_new)
        loss = self.loss_function(out, y)
        dl_dstate_old, dl_dtheta_direct = grad(loss, (state_vec_old, self.theta), retain_graph=True)

        if self.s_toupee is None:
            self.s_toupee = Variable(torch.zeros(*state_vec_old.size()))  # (batch_size, state_dim)
            self.theta_toupee = Variable(torch.zeros(*self.theta.size()))  # (n_params, )

        indirect_grad = (dl_dstate_old*self.s_toupee).sum()*self.theta_toupee
        pseudograds = indirect_grad + dl_dtheta_direct

        # Do ForwardDiff pass
        state_old_perturbed = state_vec_old.cleave(state_vec_old + self.s_toupee * self.epsilon_perturbation)#.detach()
        state_vec_new_perturbed = MergedVariable.join(self.forward_update_module(x, state_old_perturbed)[1])
        state_deriv_in_direction_s_toupee = (state_vec_new_perturbed - state_vec_new)/self.epsilon_perturbation

        nus = Variable(torch.round(torch.rand(*state_vec_old.size()))*2-1)

        # Backprop nus through the rnn
        direct_theta_toupee_contribution, = grad(outputs=state_vec_new, inputs=self.theta, grad_outputs=nus)

        rho_0 = torch.sqrt((self.theta_toupee.norm() + self.epsilon_stability)/(state_deriv_in_direction_s_toupee.norm() + self.epsilon_stability))
        rho_1 = torch.sqrt((direct_theta_toupee_contribution.norm() + self.epsilon_stability)/(nus.norm() + self.epsilon_stability))
        self.theta.grad = pseudograds
        if self.learning_rate_generator is not None:
            set_optimizer_learning_rate(self.optimizer, next(self.learning_rate_generator))
        self.optimizer.step()

        self.s_toupee = (rho_0*state_deriv_in_direction_s_toupee + rho_1*nus).detach()
        self.theta_toupee = (self.theta_toupee/rho_0 + direct_theta_toupee_contribution/rho_1).detach()
        self._state = state_new
        return out

    def get_state(self):
        return clone_em((self.theta, self._state, self.s_toupee, self.theta_toupee))

    def set_state(self, state):
        theta, self._state, self.s_toupee, self.theta_toupee = state
        self.theta.data[:] = theta.data
