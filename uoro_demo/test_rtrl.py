from recurrent_problems.datasets.anbn import get_an_bn_prediction_dataset
from spiking_experiments.farnn.recurrent_learners.rtrl import RTRL


def test_rtrl(n_training_steps=1000, span = 3):
    """
    Simple task... a^n b^n for n=3.  This has got to work.
    :return:
    """

    x_train, y_train = get_an_bn_prediction_dataset(n_training_steps=n_training_steps, k=span, l=span, onehot_inputs=True, onehot_target=False)

    n_in = n_out = 3


    net = RTRL(
        f
    )

    net = get_predictor(n_in=n_in, n_hid=n_hid, n_out=n_out, predictor_type=predictor_type, rnn_type=rnn_type, loss=loss,
        optimizer=optimizer, learning_rate=learning_rate, predictor_options=predictor_options, forward_options=forward_options
        )

    train_test_errors = train_online_network(
        model = net,
        dataset = dataset,
        n_tests= n_splits,
        error_func=error_func,
        batchify=True,
        test_online=True,
        online_test_reporter = 'recent',
        )


