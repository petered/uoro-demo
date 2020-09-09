from artemis.experiments.experiment_management import load_record_results
from artemis.general.ezprofile import EZProfiler
from matplotlib import pyplot as plt
import numpy as np
from artemis.experiments.experiment_record_view import separate_common_args
from artemis.general.should_be_builtins import bad_value
from artemis.plotting.pyplot_plus import get_lines_color_cycle

from src.artemis.artemis.ml.tools.processors import RunningAverage, RecentRunningAverage


def plot_learning_curve(train_test_errors, scale='loglog'):

    cumulative_loss = RunningAverage.batch(train_test_errors['online_errors'])
    recent_loss = RecentRunningAverage.batch(train_test_errors['online_errors'])

    plt_func = plt.loglog if scale =='loglog' else plt.plot if scale=='normal' else bad_value(scale)

    plt_func(recent_loss, label = 'Recent Loss')
    plt_func(cumulative_loss, label = 'Cumulative Loss')
    plt.xlabel('t')
    plt.ylabel('error')
    plt.legend()
    plt.grid()


def plot_multiple_learning_curves(curves, labels = None, xscale='log', yscale='linear', subssample_threshold = None):

    if labels is None:
        labels = [None]*len(curves)

    for c, (label, loss) in zip(get_lines_color_cycle(), zip(labels, curves)):

        cumulative_loss = RunningAverage.batch(loss)
        recent_loss = RecentRunningAverage.batch(loss)
        # if subssample_threshold is not None and len()
        if subssample_threshold is not None and len(loss)>subssample_threshold:
            points = {
                'linear': lambda: np.round(np.linspace(0, len(loss)-1, subssample_threshold)).astype(np.int),
                'log': lambda: np.round(np.logspace(1, np.log10(len(loss)-1), subssample_threshold)).astype(np.int),
                }[xscale]()
            plt.plot(points, recent_loss[points], alpha=0.5, color = c)
            plt.plot(points, cumulative_loss[points], label = label, color=c)
        else:
            plt.plot(recent_loss, alpha=0.5, color = c)
            plt.plot(cumulative_loss, label = label, color=c)
        plt.gca().set_xscale(xscale)
        plt.gca().set_yscale(yscale)
        plt.xlabel('t')
        plt.ylabel('error')
        plt.legend()
        plt.grid()


def compare_learning_curves(records):

    results_dict = load_record_results(records, err_if_no_result=False)
    _, argdiff = separate_common_args(records, return_dict=True)
    plt.figure()
    plot_multiple_learning_curves([v['online_errors'] for v in results_dict.values()], labels=argdiff.values())
    plt.show()