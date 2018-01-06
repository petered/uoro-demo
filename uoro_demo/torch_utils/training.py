from collections import OrderedDict
from functools import partial

import itertools

import logging
import numpy as np
import torch
from torch.autograd import grad
import time

from artemis.general.display import sensible_str, deepstr
from artemis.general.nested_structures import seqstruct_to_structseq, nested_map
from artemis.general.progress_indicator import ProgressIndicator
from artemis.general.should_be_builtins import bad_value
from artemis.ml.tools.processors import RunningAverage, RecentRunningAverage
from uoro_demo.torch_utils.torch_helpers import numpy_struct_to_torch_struct, torch_loop


def percent_argmax_correct(a, b):

    # if a.dim()==2:
    _, a = torch.max(a, dim=-1)
    # else:
    #     assert a.dim() == 1
    # if b.dim()==2:
    #     _, b = torch.max(b, dim=1)
    # else:
    #     assert b.dim() == 1

    return torch.eq(a, b).double().mean()*100


log2 = np.log(2.)
_str_to_layerclass = {
    'tanh': torch.nn.Tanh,
    'relu': torch.nn.ReLU,
    }
_str_to_loss_func = {
    'mse': torch.nn.MSELoss,
    'l1': torch.nn.L1Loss,
    'xe': torch.nn.CrossEntropyLoss,
    'nll': torch.nn.NLLLoss,
    'xebits': lambda xe=torch.nn.CrossEntropyLoss(): lambda a, b: xe(a, b)/log2,
    'percent_argmax_correct': lambda: percent_argmax_correct,
}
_str_to_rnn_class = {
    'lstm': torch.nn.LSTM,
    'elman': torch.nn.RNN,
    'gru': torch.nn.GRU,
}


def create_activation_layer(name):
    return _str_to_layerclass[name]()


def create_loss_function(name):
    return _str_to_loss_func[name]()


def get_rnn_class(string):
    return _str_to_rnn_class[string]


def get_initial_hidden_state(rnn_type, n_hidden, batch_size=1, n_layers=1):
    return \
        (torch.autograd.Variable(torch.zeros(n_layers, batch_size, n_hidden)), torch.autograd.Variable(torch.zeros(n_layers, batch_size, n_hidden))) if rnn_type=='lstm' else \
        torch.autograd.Variable(torch.zeros(n_layers, batch_size, n_hidden)) if rnn_type =='elman' else \
        bad_value(rnn_type)


def get_named_torch_optimizer_factory(name, learning_rate):
    return partial({
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'rmsprop': torch.optim.RMSprop,
        }[name.lower()], lr=learning_rate)


def train_online_network_checkpoints(model, dataset, checkpoint_generator = None, test_online=True, return_output = True, n_tests=0, offline_test_mode=None, online_test_reporter ='recent', error_func ='mse', batchify = False, print_every = 5):
    """

    :param model: A TrainableStatefulModule
    :param dataset: A 4-tuple of (x_train, y_train, x_test, y_test) where the first axis of each is the sample #
    :param n_tests: Number of "splits", in the training set... (where we run a full test)
    :return: train_test_errors: A tuple of (time_step, train_error, test_error)
    """
    data = numpy_struct_to_torch_struct(dataset, cast_floats='float32')
    if batchify:
        data = [x[:, None] for x in data]
    if len(data)==4:
        x_train, y_train, x_test, y_test = data
    elif len(data)==2:
        x_train, y_train = data
        x_test, y_test = [], []
    else:
        raise Exception('Expected data to be (x_train, y_train, x_test, y_test) or (x_test, y_test)')
    assert len(y_train) == len(x_train)
    assert len(x_test) == len(y_test)

    if isinstance(checkpoint_generator, tuple):
        distribution = checkpoint_generator[0]
        if distribution == 'even':
            interval, = checkpoint_generator[1:]
            checkpoint_generator = (interval*i for i in itertools.count(1))
        elif distribution == 'exp':
            first, growth = checkpoint_generator[1:]
            checkpoint_generator = (first*i*(1+growth)**(i-1) for i in itertools.count(1))
        else:
            raise Exception("Can't make a checkpoint generator {}".format(checkpoint_generator))

    if isinstance(error_func, basestring):
        error_func = create_loss_function(error_func)

    n_training_samples = len(x_train)
    test_iterations = [int(n_training_samples * i / float(n_tests - 1)) for i in xrange(0, n_tests)]

    initial_state = model.get_state()

    results = SequentialStructBuilder()

    if test_online:
        loss_accumulator = RunningAverage() if online_test_reporter=='cum' else RecentRunningAverage() if online_test_reporter=='recent' else lambda x: x if online_test_reporter is None else bad_value(online_test_reporter)

    t_start = time.time()
    next_checkpoint = float('inf') if checkpoint_generator is None else next(checkpoint_generator)
    err = np.nan
    pi = ProgressIndicator(n_training_samples+1, update_every=(print_every, 'seconds'), show_total=True,
        post_info_callback=lambda: 'Iteration {} of {}. Online {} Error: {}'.format(t, len(x_train), online_test_reporter, err))
    for t in xrange(n_training_samples+1):

        if offline_test_mode is not None and t in test_iterations:
            training_state = model.get_state()
            model.set_state(initial_state)

            if offline_test_mode == 'full_pass':
                y_train_guess = torch_loop(model, x_train[:t]) if t > 0 else None
                if t<len(x_train)-1:
                    y_middle_guess = torch_loop(model, x_train[t:None])
                y_test_guess = torch_loop(model, x_test)

                train_err = error_func(_flatten_first_2(y_train_guess), _flatten_first_2(y_train[:t])).data.numpy()[0] if y_train_guess is not None else np.nan
                test_err = error_func(_flatten_first_2(y_test_guess), _flatten_first_2(y_test)).data.numpy()[0]

                # train_err, test_err = tuple((y_guess - y_truth).abs().sum().data.numpy() for y_guess, y_truth in [(y_train_guess, y_train[:t] if t>0 else torch.zeros(2, 2, 2)/0), (y_test_guess, y_test)])
                print 'Iteration {} of {}: Training: {:.3g}, Test: {:.3g}'.format(t, len(x_train), train_err, test_err)
                results['offline_errors']['t'].next = t
                results['offline_errors']['train'].next = train_err
                results['offline_errors']['test'].next = test_err

            elif offline_test_mode == 'cold_test':
                y_test_guess = torch_loop(model, x_test)
                test_err = error_func(_flatten_first_2(y_test_guess), _flatten_first_2(y_test)).data.numpy()[0]
                print 'Iteration {} of {}: Test: {:.3g}'.format(t, len(x_train), test_err)
                results['offline_errors']['t'].next = t
                results['offline_errors']['test'].next = test_err
            else:
                raise Exception('No test_mode: {}'.format(offline_test_mode))
            model.set_state(training_state)

        if t<n_training_samples:
            out = model.train_it(x_train[t], y_train[t])
            if return_output:
                results['output'].next = out.data.numpy()[0]
            if test_online:
                # print 'Out: {}, Correct: {}'.format(np.argmax(out.data.numpy(), axis=1), torch_str(y_train[t]))
                this_loss = error_func(out, y_train[t]).data.numpy()[0]
                err = loss_accumulator(this_loss)
                results['online_errors'].next = this_loss
                if online_test_reporter is not None:
                    results['smooth_online_errors'][online_test_reporter].next = err
                # if t in test_iterations:
                #     print('Iteration {} of {} Online {} Error: {}'.format(t, len(x_train), online_test_reporter, err))

        pi()
        if t>=next_checkpoint or t==n_training_samples:
            results['checkpoints'].next = {'iter': t, 'runtime': time.time()-t_start}
            yield results.to_structseq(as_arrays=True)
            next_checkpoint = next(checkpoint_generator)
            # yield nested_map(lambda x: np.array(x), results, is_container_func=lambda x: isinstance(x, dict))

    # yield results.to_structseq(as_arrays=True)
    # yield nested_map(lambda x: np.array(x), results, is_container_func=lambda x: isinstance(x, dict))


def train_online_network(*args, **kwargs):

    for result in train_online_network_checkpoints(*args, **kwargs):
        pass

    return result


def _flatten_first_2(x):
    size = x.size()
    return x.view(size[0]*size[1], *size[2:])


def grad_with_disconnected_zero(outputs, inputs, retain_graph=False, **kwargs):

    grads = []

    inputs = list(inputs)

    for i, p in enumerate(inputs):
        try:
            p_grad, = grad(outputs, p, retain_graph=i<len(inputs)-1 or retain_graph, **kwargs)
        except RuntimeError as err:
            if err.message in ('differentiated input is unreachable', 'One of the differentiated Variables appears to not have been used in the graph'):
                p_grad = torch.autograd.Variable(torch.zeros(p.size()))
            else:
                raise
        grads.append(p_grad)
    return grads


def jacobian(out_vec, in_vec):
    """
    :param out_vec:
    :param in_vec:
    :return: An (len(out_vec), len(in_vec)) jacobian
    """
    assert len(out_vec.size())==1
    assert len(in_vec.size())==1
    return torch.cat([grad(o, in_vec, retain_graph=True)[0][None] for o in out_vec], dim=0)


def set_optimizer_learning_rate(optimizer, learning_rate):
    for pg in optimizer.param_groups:
        pg['lr'] = learning_rate



class SequentialStructBuilder(object):
    """
    A convenient structure for storing results.
    (Obselete.. use Duck instead)
    """
    def __init__(self, struct=None):
        self._struct = struct  # An OrderedDict<string: list<obj>>

    def __getitem__(self, key):
        if self._struct is None:
            self._struct = OrderedDict()
        if key not in self._struct:
            self._struct[key] = SequentialStructBuilder()
        return self._struct[key]

    def __setitem__(self, key, value):
        try:
            self._struct[key] = value
        except TypeError:
            if self._struct is None:
                self._struct = OrderedDict()
                self._struct[key] = value
            else:
                raise TypeError('{} has already been defined as a list, but you are trying to set a key "{}" as if it were a dict. '.format(self, key))

        # else:
        #     assert isinstance(self._struct, OrderedDict)
        # self._struct[key] = value

    def __iter__(self):
        assert self.is_sequence is True
        return (s for s in self._struct)

    def __len__(self):
        return len(self._struct) if self._struct is not None else 0

    @property
    def is_sequence(self):
        if isinstance(self._struct, OrderedDict):
            return False
        elif isinstance(self._struct, list):
            return True
        else:
            assert self._struct is None
            return None

    def items(self):
        assert self.is_sequence is False, 'You can only call "items" when the Structure has been defined as a dict'
        return self._struct.items()

    def open_next(self):
        """
        Add a new element to this sequence, so that future calls to last return this element.
        :return:
        """
        new_chapter = SequentialStructBuilder()
        self.next = new_chapter
        return new_chapter

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, sensible_str(self._struct))

    def description(self):
        print self.__class__.__name__ +":" + deepstr(self.to_struct())

    def values(self):
        return self if self.is_sequence else self._struct.values()

    @property
    def next(self):
        logging.warn("Warning: next should only be set, not gotten")
        return None

    @property
    def each_next(self):
        logging.warn("Warning: next should only be set, not gotten")
        return None

    @each_next.setter
    def each_next(self, dict_of_items):
        assert self.is_sequence is not True, 'each_next can only be applied to a dict-type object'
        if self.is_sequence is None:
            for k, v in dict_of_items.items():
                self[k] = []
        for k, v in dict_of_items.items():
            assert k in self._struct, "You previously called each_next with a different data structure."
            self[k].append(v)

    @property
    def first(self):
        assert self.is_sequence is not None, "Structure is empty"
        return self._struct[0] if self.is_sequence else self._struct.values()[0]

    @property
    def last(self):
        assert self.is_sequence is not None, "Structure is empty"
        return self._struct[-1] if self.is_sequence else self._struct.values()[-1]

    def to_array(self):
        items = [item.to_array() if isinstance(item, SequentialStructBuilder) else item for item in self.values()]
        return np.array(items)

        #     return np.array([item.to_array() if isinstance(item, SequentialStructBuilder) else item for item in self])
        # else:
        #     return np.array(self.to_struct().values())
        # assert self.is_sequence, 'Can only call to_array when the SequentialStructBuilder has been used as a sequence.  It has not.'

    def is_arrayable(self):
        return self.is_sequence and all(isinstance(s, (int, list, float, np.ndarray)) or (isinstance(s, SequentialStructBuilder) and s.is_arrayable()) for s in self)

    def to_struct_arrays(self):
        """
        Recursively convert this structure into ndarrays wherever possible.
        :return: A nested structure with arrays at the leaves.
        """
        return self.to_array() if self.is_arrayable() else self.map(lambda el: el.to_struct_arrays() if isinstance(el, SequentialStructBuilder) else el).to_struct()

    @next.setter
    def next(self, val):
        try:
            self._struct.append(val)
        except AttributeError:
            if self._struct is None:
                self._struct = []
                self._struct.append(val)
            else:
                raise TypeError('{} has already been defined as a dict, but you are trying to append an element to it as if it were a list. '.format(self))

    def map(self, func):
        if self.is_sequence is True:
            return SequentialStructBuilder([func(x) for x in self._struct])
        elif self.is_sequence is False:
            return SequentialStructBuilder(OrderedDict((name, func(val)) for name, val in self._struct.items()))
        else:
            return None

    def to_struct(self):
        """
        Recursively transform this object into a nested struct.
        :return: A nested struct of OrderedDicts and lists
        """
        return self.map(lambda v: v.to_struct() if isinstance(v, SequentialStructBuilder) else v)._struct

    def to_structseq(self, as_arrays=False):
        structs = self.to_struct()
        return nested_map(lambda s: seqstruct_to_structseq(s, as_arrays=as_arrays) if isinstance(s, list) else s, structs, is_container_func = lambda x: isinstance(x, dict))

    def to_seqstruct(self):
        structs = self.to_struct()
        return nested_map(lambda s: seqstruct_to_structseq(s) if isinstance(s, dict) else s, structs, is_container_func = lambda x: isinstance(x, list))