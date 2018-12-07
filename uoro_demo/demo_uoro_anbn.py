import itertools
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from tabulate import tabulate
from artemis.experiments.decorators import ExperimentFunction
from artemis.experiments.experiment_management import load_record_results
from artemis.experiments.experiment_record_view import show_record
from artemis.general.should_be_builtins import uniquify_duplicates
from artemis.general.test_mode import is_test_mode
from uoro_demo.anbn import get_an_bn_prediction_dataset
from uoro_demo.online_experiment_helpers import plot_learning_curve, \
    plot_multiple_learning_curves

from uoro_demo.online_predictors import get_online_predictor, OnlinePredictorTypes as OPT
from uoro_demo.torch_utils.training import train_online_network_checkpoints

"""
Compare UORO and RTRL on the a^n b^n task.

To start, run this file and enter "run 0" for UORO or "run 1" to run RTRL.

After running for a while you can compare results, with "compare 0,1"
"""


def _get_correct_endings(predictions, y_train):
    """
    :param predictions: (n_samples, )
    :param y_train: (n_samples, )
    :return:
    """
    y_train = y_train[:len(predictions)]
    end_ixs = np.where((y_train[:-1]==0) & (y_train[1:]==1))[0]  # End is where y_train[t]==0 and y_train[t+1]==1
    assert np.all(y_train[end_ixs]==0)
    correct_endings = (predictions[end_ixs-1]==2) & (predictions[end_ixs]==0)
    return correct_endings


def show_this_record(record, scale='normal'):
    show_record(record)

    result = record.get_result()
    args = record.get_args()

    n_training_steps = result['checkpoints', -1, 'iter'] if 'checkpoints' in result else args['n_training_steps']

    data = get_an_bn_prediction_dataset(n_training_steps=n_training_steps, n_test_steps=args['n_training_steps'], k=args['k'], l=args['l'], onehot_inputs=True, onehot_target=False)
    y_train = data[1]
    print('Prediction: '+ str(np.argmax(result['output', -100:], axis=1)).replace('\n', ''))
    print('Truth:      '+str(y_train[-100:]).replace('\n', ''))

    # Now, just find how it did at predicting the 0's at the end of sequences.
    correct_endings = _get_correct_endings(predictions=np.argmax(result['output'], axis=1), y_train=y_train)
    result_filtered = result.copy()
    result_filtered['online_errors'] = correct_endings

    plt.figure()
    plt.subplot(2,1,1)
    plot_learning_curve(result, scale=scale)
    plt.subplot(2,1,2)
    plot_learning_curve(result_filtered, scale=scale)
    plt.title('Success on only end of sequence')
    plt.show()


def compare_records(records, subsample_threshold = 10000):
    """
    Make plots comparing convergence across records.
    :param records:
    :param subsample_threshold:
    :return:
    """

    results_dict = load_record_results(records, err_if_no_result=False, index_by_id=True)
    all_args = [rec.get_args() for rec in records]
    names = uniquify_duplicates([args['predictor_type'] for args in all_args])
    try:
        print(tabulate([[name, result['checkpoints', -1, 'iter']/result['checkpoints', -1, 'runtime']] for result, name in zip(results_dict.values(), names)], headers=['Name', 'Iter/s']))
    except KeyError:
        print("You have some old records that don't have checkpoints")
    n_training_steps = max(result['checkpoints', -1, 'iter'] if 'checkpoints' in result else args['n_training_steps'] for result, args in zip(results_dict.values(), all_args))
    args = records[0].get_args()
    data = get_an_bn_prediction_dataset(n_training_steps=n_training_steps, n_test_steps=args['n_training_steps'], k=args['k'], l=args['l'], onehot_inputs=True, onehot_target=False)
    y_train = data[1]
    errors = [v['online_errors'].to_array() for v in results_dict.values()]
    correct_ends = [_get_correct_endings(np.argmax(v['output'], axis=1), y_train) for v in results_dict.values()]

    plt.figure()
    plt.subplot(2, 1, 1)
    plot_multiple_learning_curves(errors, labels=names, subssample_threshold=subsample_threshold)
    plt.subplot(2, 1, 2)
    plot_multiple_learning_curves(correct_ends, labels=names, subssample_threshold=subsample_threshold)
    plt.show()


@ExperimentFunction(show=show_this_record, compare=compare_records, is_root=True)
def demo_anbn_prediction(
        n_training_steps = 100000,
        k=1,
        l=32,
        n_hid=64,
        predictor_type ='uoro',
        predictor_options = {},
        rnn_type='GRUPredictor',
        rnn_options = dict(bias=False),
        optimizer='adam',
        learning_rate = 0.001,
        n_splits=10,
        error_func ='xebits',
        loss='xe',
        alpha = 3e-2,
        seed=1234,
        ):
    """
    :param n_training_steps:
    :param k: k parameter of anan dataset... minimum sequence length
    :param l: l parameter of anan dataset... maximum sequence length
    :param n_hid:
    :param predictor_type: Either 'rtrl' or 'ouro'
    :param predictor_options: Additional kwargs to pass to predictor constructor
    :param rnn_type: 'gru', 'lstm', or 'elman'  (note.... some may be broken now)
    :param rnn_options: Additional kwargs to pass to RNN constructor
    :param optimizer: Optimizer type 'sgd', 'adam', 'rmsprop'...
    :param learning_rate: Initial learning rate
    :param n_splits: (ignore this?)
    :param error_func: String identifying function to report error.  'xe' is cross-entropy, 'xebits' is cross-entropy in bits, 'mse', ...
    :param loss: String identifying function to compute loss
    :param alpha: Controls decay of learning rate lr[t] = lr/(1+alpha*sqrt(t))
    :param seed: Random seed
    :yield: A data structure containing learning curve data.  Inspect it if you need it, or just inspect results using 'compare'
        command in experiment ui.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if is_test_mode():
        n_training_steps=10

    predictor_options = predictor_options.copy()
    if alpha is not None:
        predictor_options['learning_rate_generator'] = (learning_rate/(1.+alpha*np.sqrt(t)) for t in itertools.count(0))

    x_train, y_train = get_an_bn_prediction_dataset(n_training_steps=n_training_steps, n_test_steps=None, k=k, l=l, onehot_inputs=True, onehot_target=False)
    n_in = n_out = 3

    net = get_online_predictor(n_in=n_in, n_hid=n_hid, n_out=n_out, predictor_type=predictor_type, rnn_type=rnn_type, rnn_options=rnn_options, loss=loss,
                               optimizer=optimizer, learning_rate=learning_rate, predictor_options=predictor_options,
                               )

    for result in train_online_network_checkpoints(
            model = net,
            dataset = (x_train, y_train),
            n_tests= n_splits,
            error_func=error_func,
            batchify=True,
            test_online=True,
            online_test_reporter = 'recent',
            checkpoint_generator=('exp', 1000, 0.1)
            ):
        print('Yielding Result at {} iterations.'.format(result['checkpoints', -1, 'iter']))
        yield result


X1 = demo_anbn_prediction.add_root_variant('easy', k=4, l=4, n_training_steps=10000)
X2 = demo_anbn_prediction.add_root_variant('medium', k=1, l=4, n_training_steps=40000)
X3 = demo_anbn_prediction.add_root_variant('hard', k=1, l=8, n_training_steps=40000)
X4 = demo_anbn_prediction.add_root_variant('insane', k=1, l=32, n_training_steps=int(1e7))

for XX in [X1, X2, X3, X4]:
    for predictor_type in [
            OPT.UORO,
            OPT.RTRL,
            ]:
        XX.add_variant(predictor_type, predictor_type=predictor_type)

if __name__ == '__main__':

    demo_anbn_prediction.browse(display_format='flat', filterexp='has:insane', filterrec = 'result@last')
    # You can run both experiments in parallel with 'run all -p'
    # Later, when records have been saved, you can view results with 'compare all'
    # You can also just run demo_anbn_prediction() alone if you don't want to bother with saving results.
