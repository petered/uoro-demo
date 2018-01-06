import torch
from torch.nn import Linear, Sigmoid, Tanh
from torch import sigmoid, tanh
from artemis.general.should_be_builtins import bad_value
from uoro_demo.torch_utils.interfaces import RecurrentStatelessModule
from uoro_demo.torch_utils.training import get_rnn_class


class StatelessPredictorRNN(RecurrentStatelessModule):

    def __init__(self, n_in, n_hid, n_out, rnn_type='elman', output_rep='linear', freeze_rnn = False, bias=True):

        super(StatelessPredictorRNN, self).__init__()
        self.rnn = get_rnn_class(rnn_type)(n_in, n_hid)
        self.n_hid=n_hid
        self.n_out = n_out
        self.rnn_type = rnn_type
        self.output_layer = {
            'linear': lambda: torch.nn.Linear(n_hid, n_out, bias = bias),
            }[output_rep]()
        self.freeze_rnn = freeze_rnn

        if freeze_rnn:
            it_happened = False
            for param_name, p in self.rnn.named_parameters():
                if param_name.startswith('weight_hh') or param_name.startswith('bias_hh'):
                    p.requires_grad = False
                    it_happened = True
            assert it_happened

    #
    # def parameters(self):
    #     if self.freeze_rnn:
    #         # Note this also disables the input-to-state parameters, but that's ok for now.
    #         rnn_parameters = list(self.rnn.parameters())
    #         return (p for p in super(StatelessPredictorRNN, self).parameters() if p not in rnn_parameters)
    #     else:
    #         return (p for p in super(StatelessPredictorRNN, self).parameters())

    def forward(self, x, prev_state):
        """
        :param x: A (batch_size, n_dim) input
        :param prev_state: A representation of the previous hidden state: (
        :return: A (batch_size, n_out) output
        """
        hidden_history, hidden_state = self.rnn(x[None], prev_state)
        prediction = self.output_layer(hidden_history[-1, :, :])  # (last_step, )
        return prediction, hidden_state

        # _, hidden_state = self.rnn(x[None], prev_state)
        # hidden_history, _ = self.rnn(x[None], prev_state)
        # prediction = self.output_layer(hidden_history[-1, :, :])  # (last_step, )
        # return prediction, hidden_state

    def get_initial_state(self, x_init):
        assert x_init.dim() == 2, 'x_init should have 2 dimensions: (batch_size, n_dims).  Its shape is {}'.format(x_init.size())

        return torch.autograd.Variable(torch.zeros(1, len(x_init), self.n_hid), requires_grad=True) if self.rnn_type in ('elman', 'gru') else \
            (torch.autograd.Variable(torch.zeros(1, len(x_init), self.n_hid), requires_grad=True), torch.autograd.Variable(torch.zeros(1, len(x_init), self.n_hid), requires_grad=True)) if self.rnn_type == 'lstm' else \
            bad_value(self.rnn_type)


class GRUPredictor(RecurrentStatelessModule):

    def __init__(self, n_in, n_hid, n_out, bias = True):
        super(GRUPredictor, self).__init__()
        self.n_hid = n_hid
        self.f_xz = Linear(n_in, n_hid, bias=bias)
        self.f_xr = Linear(n_in, n_hid, bias=bias)
        self.f_xh = Linear(n_in, n_hid, bias=bias)
        self.f_hz = Linear(n_hid, n_hid, bias=False)
        self.f_hr = Linear(n_hid, n_hid, bias=False)
        self.f_hrh = Linear(n_hid, n_hid, bias=False)
        self.f_hy = Linear(n_hid, n_out, bias=bias)

    def forward(self, x, h_old):
        z = sigmoid(self.f_xz(x) + self.f_hz(h_old))
        r = sigmoid(self.f_xr(x) + self.f_hr(h_old))
        h = z * h_old + (1-z)* tanh(self.f_xh(x) + self.f_hrh(r*h_old))
        y = self.f_hy(h)
        return y, h

    def get_initial_state(self, x_init):
        return torch.autograd.Variable(torch.zeros(len(x_init), self.n_hid), requires_grad=True)