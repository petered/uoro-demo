
import numpy as np

from artemis.general.mymath import onehotvector
from artemis.ml.tools.processors import OneHotEncoding


def get_an_bn_data(n_samples, k=1, l=32, seed=1234, onehot=False):
    """

    :param k:
    :param l:
    :return:
    """

    rng = np.random.RandomState(seed)

    x = np.zeros(n_samples, dtype=int)

    i=0
    while i<n_samples:
        n = rng.randint(low=k, high=l+1)
        x[i:i+n] = 1
        x[i+n+1: i+2*n+1] = 2
        i += 2*n + 2

    if onehot:
        x = onehotvector(x, 3)

    return x


def get_an_bn_prediction_dataset(n_training_steps, n_test_steps=None, k=1, l=32, seed=1234, onehot_inputs=False, batchify = False, onehot_target=False):

    x = get_an_bn_data(n_samples=n_training_steps+n_test_steps+1 if n_test_steps is not None else n_training_steps+1, k=k, l=l, seed=seed, onehot=False)

    if batchify:
        x = x[:, None]

    y = x[1:]
    x = x[:-1]

    if onehot_inputs:
        x = OneHotEncoding(3, dtype='float32')(x)
    if onehot_target:
        y = OneHotEncoding(3, dtype='float32')(y)

    if n_test_steps is None:
        return x, y
    else:
        return x[:n_training_steps], y[:n_training_steps], x[n_training_steps:], y[n_training_steps:]


def get_correct_anbn_predictions(predictions, y_train, return_indeces = False):
    """
    Return a
    :param predictions: (n_samples, )
    :param y_train: (n_samples, )
    :return: A boolean vector indicating whether each sequence-ending was correctly identified.
    """
    y_train = y_train[:len(predictions)]
    end_ixs = np.where((y_train[:-1]==0) & (y_train[1:]==1))[0]  # End is where y_train[t]==0 and y_train[t+1]==1
    assert np.all(y_train[end_ixs]==0)
    correct_endings = (predictions[end_ixs-1]==2) & (predictions[end_ixs]==0)
    return correct_endings if not return_indeces else (end_ixs, correct_endings)



if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_an_bn_prediction_dataset(100, split_point=2./3, k=1, l=4)
    print ('x_train: {}'.format(list(x_train)))
    print ('y_train: {}'.format(list(y_train)))
    print ('x_test:  {}'.format(list(x_test)) )
    print ('y_test:  {}'.format(list(y_test)) )
