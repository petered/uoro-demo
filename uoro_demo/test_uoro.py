import numpy as np
import torch
from torch.autograd import Variable
from argparse import Namespace
from artemis.general.ezprofile import EZProfiler
from artemis.ml.tools.processors import RunningAverage
from artemis.plotting.easy_plotting import funplot
from recurrent_problems.datasets.anbn import get_an_bn_prediction_dataset
from recurrent_problems.datasets.synthetic_rnn_data import generate_synthetic_rnn_data
from spiking_experiments.farnn.recurrent_learners.predictor_funcs import StatelessPredictorRNN
from spiking_experiments.farnn.recurrent_learners.rtrl import RTRL
from spiking_experiments.farnn.recurrent_learners.uoro import UOROVec
from spiking_experiments.farnn.recurrent_learners._uoro_deprecated import UORO_Deprecated
from spiking_experiments.farnn.torch_utils.torch_helpers import torch_str
from spiking_experiments.farnn.torch_utils.training import get_named_torch_optimizer_factory, \
    numpy_struct_to_torch_struct, torch_loop, train_online_network


def vequal(v1, v2):
    return int((v1!=v2).sum()[0].data.numpy())==0


def test_uoro_runs(batch_size=1, n_in=5, n_hid=10, n_out=3):

    x = torch.autograd.Variable(torch.randn(batch_size, n_in))
    y = torch.autograd.Variable(torch.randn(batch_size, n_out))

    net = UORO_Deprecated(
        forward_update_module=StatelessPredictorRNN(n_in=n_in, n_hid=n_hid, n_out=n_out, rnn_type='elman'),
        loss='mse',
        optimizer_constructor=get_named_torch_optimizer_factory('sgd', 0.01)
        )

    initial_state = net.get_state()
    y_initial_guess = net(x)
    assert y_initial_guess.size() == (batch_size, n_out)

    y_second_guess = net(x)
    assert not vequal(y_second_guess, y_initial_guess)

    net.set_state(initial_state)
    y_third_guess = net(x)
    assert vequal(y_third_guess, y_initial_guess)

    net.train_it(x, y)

    net.set_state(initial_state)
    y_fouth_guess = net(x)
    assert not vequal(y_fouth_guess, y_initial_guess)

    net.train_it(x, y)


def test_uoro_works(n_steps = 40, n_in=5, n_hid=10, n_out=3):

    xs, ys = generate_synthetic_rnn_data(n_steps = n_steps, n_in=n_in, n_hidden=n_hid, n_out=n_out, rng=123)

    torch.manual_seed(1234)

    xs, ys = numpy_struct_to_torch_struct((xs[:, None, :].astype('float32'), ys[:, None, :].astype('float32')))

    net = UORO_Deprecated(
        forward_update_module=StatelessPredictorRNN(n_in=n_in, n_hid=n_hid, n_out=n_out, rnn_type='elman'),
        loss='mse',
        optimizer_constructor=get_named_torch_optimizer_factory('sgd', 0.1),
        )

    initial_state = net.get_state()
    initial_guess = torch_loop(net, xs)

    net.set_state(initial_state)
    torch_loop(net.train_it, xs, ys)

    net.set_state(initial_state)
    final_guess = torch_loop(net, xs)

    initial_error = ((initial_guess-ys)**2).mean()
    final_error = ((final_guess-ys)**2).mean()

    print 'Initial Error: {}\nFinal Error: {}'.format(torch_str(initial_error), torch_str(final_error))
    assert 0.075<initial_error.data[0]<0.076
    assert 0.012<final_error.data[0]<0.013


def test_uoro_anbn(n_training_steps = 2000, n_test_steps = 1000, k=4, l=4, n_in=3, n_hid=10, n_out=3,
        rnn_type='gru', optimizer='sgd', learning_rate = 0.01, n_splits=10, loss = 'xe',
        error_func ='xebits', predictor_options = {}, seed=1234):
    """
    Assert that uoro passes anbn perfectly
    :param n:
    :return:
    """

    torch.manual_seed(seed)

    dataset = get_an_bn_prediction_dataset(n_training_steps=n_training_steps, n_test_steps=n_test_steps, k=k, l=l, onehot_inputs=True, onehot_target=False)
    x_train, y_train, x_test, y_test = dataset

    assert n_in==3 and n_out==3
    net = UORO_Deprecated(
            forward_update_module=StatelessPredictorRNN(n_in=n_in, n_hid=n_hid, n_out=n_out, rnn_type=rnn_type, output_rep='linear'),
            loss=loss,
            optimizer_constructor=get_named_torch_optimizer_factory(optimizer, learning_rate),
            nu_policy='random'
            )
    train_test_errors = train_online_network(
        model = net,
        dataset = dataset,
        error_func=error_func,
        batchify=True,
        test_online=True,
        online_test_reporter = 'recent'
        )


    # [-1.65793109  5.06373358 -3.15905261]  # For random
    # [-1.65853739  5.06332302 -3.15808988]  # For orthogonal
    # [-1.65978789  5.06213999 -3.15585041]  # For fixed nu vector
    # [-1.65853739  5.06332302 -3.15808988]  # Fixed vector*1000

    # [-0.7433579  -0.02558513  0.06219301] with sgd
    # not 100% deterministic

    print train_test_errors['output'][-1]

    last_100_error = (np.argmax(train_test_errors['output'][-100:], axis=1)==y_train[-100:]).mean()
    print last_100_error

    return train_test_errors


# def test_uoro_same_as_uorovec(n_in = 5, n_hid=4, n_out=3, n_steps = 10):
def _compare_recurrent_nets(net_1, net_2, n_in = 20, n_out=10, n_steps = 1000, seed = 1234, expect_same=True):

    torch.manual_seed(seed)
    xy_pairs = [(Variable(torch.randn(6, n_in)), Variable(torch.randn(6, n_out))) for t in xrange(n_steps)]

    torch.manual_seed(seed)
    with EZProfiler('Net 1'):
        out_1 = [net_1.train_it(x, y) for (x, y) in xy_pairs]

    torch.manual_seed(seed)
    with EZProfiler('Net 2'):
        out_2 = [net_2.train_it(x, y) for (x, y) in xy_pairs]

    for t, (out_sep_, out_joint_) in enumerate(zip(out_1, out_2)):

        max_divergence = (out_sep_-out_joint_).abs().max().data.numpy()[0]

        if expect_same:
            assert max_divergence==0, 'Networks diverged at step {}.  Max divergence: {}'.format(t, max_divergence)
        else:
            print 'Max Difference at iteration {}: {}'.format(t, max_divergence)


def test_uoro_same_as_uorovec(n_in = 20, n_hid=20, n_out=10, n_steps = 1000):

    torch.manual_seed(1234)
    net_sep = UORO_Deprecated(
        forward_update_module=StatelessPredictorRNN(n_in=n_in, n_hid=n_hid, n_out=n_out, rnn_type='elman'),
        loss='mse',
        optimizer_constructor=get_named_torch_optimizer_factory('sgd', 0.1),
        )
    torch.manual_seed(1234)
    net_joint = UOROVec(
        forward_update_module=StatelessPredictorRNN(n_in=n_in, n_hid=n_hid, n_out=n_out, rnn_type='elman'),
        loss='mse',
        optimizer_factory=get_named_torch_optimizer_factory('sgd', 0.1),
        )
    _compare_recurrent_nets(net_sep, net_joint, n_in=n_in, n_out=n_out, n_steps=n_steps)


def test_uoro_same_as_self(n_in = 20, n_hid=20, n_out=10, n_steps = 1000):

    torch.manual_seed(1234)
    net_1 = UORO_Deprecated(
        forward_update_module=StatelessPredictorRNN(n_in=n_in, n_hid=n_hid, n_out=n_out, rnn_type='elman'),
        loss='mse',
        optimizer_constructor=get_named_torch_optimizer_factory('sgd', 0.1),
        # nu_policy = 'zero'
        )
    torch.manual_seed(1234)
    net_2 = UORO_Deprecated(
        forward_update_module=StatelessPredictorRNN(n_in=n_in, n_hid=n_hid, n_out=n_out, rnn_type='elman'),
        loss='mse',
        optimizer_constructor=get_named_torch_optimizer_factory('sgd', 0.1),
        # nu_policy = 'zero'
        )
    _compare_recurrent_nets(net_1, net_2, n_in=n_in, n_out=n_out, n_steps=n_steps)


def test_uorovec_same_as_self(n_in = 20, n_hid=20, n_out=10, n_steps = 1000):

    torch.manual_seed(1234)
    net_1 = UOROVec(
        forward_update_module=StatelessPredictorRNN(n_in=n_in, n_hid=n_hid, n_out=n_out, rnn_type='elman'),
        loss='mse',
        optimizer_factory=get_named_torch_optimizer_factory('sgd', 0.1),
        )
    torch.manual_seed(1234)
    net_2 = UOROVec(
        forward_update_module=StatelessPredictorRNN(n_in=n_in, n_hid=n_hid, n_out=n_out, rnn_type='elman'),
        loss='mse',
        optimizer_factory=get_named_torch_optimizer_factory('sgd', 0.1),
        )
    _compare_recurrent_nets(net_1, net_2, n_in=n_in, n_out=n_out, n_steps=n_steps)



def test_uoro_estimates_grad_correctly(n_steps = 5, n_runs = 100, n_in=4, n_hid=6, n_out=3, seed = 1234, plot_it = True):
    """
    Plan: run T of RTRL without learning-rate 0.  Then, on the nth step, get dl_dtheta.

    Then, starting with the same parameters, do N independent runs of UORO for T steps.

    Verify that s_tilde x t_tilde is, on average, the same as dl_dtheta.
    :return:
    """
    torch.manual_seed(seed)
    x = Variable(torch.randn(n_steps, 1, n_in))
    y = Variable(torch.LongTensor([y_%n_out for y_ in range(n_steps)])[:, None])

    rtrl = RTRL(
        forward_update_module = StatelessPredictorRNN(n_in=n_in, n_hid=n_hid, n_out=n_out, rnn_type='gru'),
        loss='xe',
        optimizer_factory=get_named_torch_optimizer_factory('sgd', 0.)
        )

    for x_, y_ in zip(x, y):
        rtrl.train_it(x_, y_)

    r = Namespace()
    r.theta, r.state, r.d_state_d_theta = rtrl.get_state()

    uoro = UOROVec(
        forward_update_module = StatelessPredictorRNN(n_in=n_in, n_hid=n_hid, n_out=n_out, rnn_type='gru'),
        loss='xe',
        optimizer_factory=get_named_torch_optimizer_factory('sgd', 0.),
        nu_policy='random'
        )

    u = Namespace()
    u.theta, u.state, u.s_toupee, u.theta_toupee = uoro.get_state()

    errors = []
    ra = RunningAverage()
    for _ in range(n_runs):
        uoro.set_state((r.theta, u.state, u.s_toupee, u.theta_toupee))
        for x_, y_ in zip(x, y):
            uoro.train_it(x_, y_)
        _, _, s_toupee, theta_toupee = uoro.get_state()

        new_average = ra(torch.ger(s_toupee, theta_toupee))

        errors.append((r.d_state_d_theta - new_average).norm().data.numpy()[0])

    print(errors)

    if plot_it:
        from matplotlib import pyplot as plt
        plt.loglog(range(1, n_runs+1), errors)
        funplot(lambda t: errors[0]/t**.5)
        plt.show()



if __name__ == '__main__':

    # test_uoro_runs()
    # test_uoro_works()
    # test_uoro_anbn()
    # test_uoro_same_as_uorovec()  # Fails
    # test_uoro_same_as_self()  # Fails
    # test_uorovec_same_as_self()  # Succeeds
    # what_are_we_waiting_for('test_uoro_same_as_uorovec()', sort_by='cumtime')
    test_uoro_estimates_grad_correctly()
