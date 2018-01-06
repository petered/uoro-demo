import torch

# from spiking_experiments.farnn.recurrent_learners.oneoff_experiments.directrnnlearner import DirectRNNLearner
# from spiking_experiments.farnn.recurrent_learners.predictor_funcs import StatelessPredictorRNN, GRUPredictor
from uoro_demo.predictor_funcs import StatelessPredictorRNN, GRUPredictor
from uoro_demo.rtrl import RTRL
from uoro_demo.torch_utils.training import get_named_torch_optimizer_factory
from uoro_demo.uoro import UOROVec
# from spiking_experiments.farnn.torch_utils.training import get_named_torch_optimizer_factory
# from spiking_experiments.farnn.truncated_bptt_predictor import TruncatedHistoryRNN


class OnlinePredictorTypes:
    UORO = 'uoro'
    RTRL = 'rtrl'


OPT = OnlinePredictorTypes


def get_online_predictor(n_in, n_hid, n_out, predictor_type, loss, optimizer, learning_rate, predictor_options = {}, rnn_type ='GRUPredictor', rnn_options = {}):
    """
    Get a trainable predictor.

    :param n_in:
    :param n_hid:
    :param n_out:
    :param predictor_type:
    :param loss:
    :param optimizer:
    :param learning_rate:
    :param rnn_type:
    :param rnn_options:
    :param output_rep:
    :param predictor_options:
    :return:
    """
    if isinstance(rnn_type, str):
        rnn_type = {cls.__name__: cls for cls in [StatelessPredictorRNN, GRUPredictor]}[rnn_type]
    else:
        assert issubclass(rnn_type, torch.nn.Module)

    forward_module = rnn_type(n_in=n_in, n_hid=n_hid, n_out=n_out, **rnn_options)

    predictor_class = {
        OPT.UORO: UOROVec,
        OPT.RTRL: RTRL,
    }[predictor_type]

    return predictor_class(
        forward_update_module = forward_module,
        loss=loss,
        optimizer_factory=get_named_torch_optimizer_factory(optimizer, learning_rate),
        **predictor_options
        )
