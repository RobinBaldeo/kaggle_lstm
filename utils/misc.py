import functools
import pdb
import time, math, itertools
import numpy as np
import pandas as pd
from typing import List, Dict
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from collections import namedtuple

score_elements = namedtuple('score_elements', 'y y_hat')


def process_time(func):
    """
    simple timer decorator
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrap_it(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        mins = round(elapsed_time / 60, 3)

        print(f"Function '{func.__name__}' took: {mins}")
        return result

    return wrap_it


def flatten_out_list(lst):
    """
    flatter out a list to unique list
    :param lst:
    :return:
    """
    return list(set(itertools.chain(*lst)))


def build_vocab(data):
    """
    convert the words to numerical mapping
    :param data: list of list of words
    :return: dic of key = words key = numbers
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}
    unique_values = list(set(itertools.chain(*data)))
    for i, j in enumerate(unique_values, start=2):
        vocab[j] = i
    return vocab


def mapping_labels(pair, labels_mapping):
    y, y_hat = pair
    return score_elements(y=labels_mapping[y], y_hat=labels_mapping[y_hat])


@process_time
def f5_score_mapping(y: list, y_hat: list, label_mapping: dict, chunk_size=5000):
    y_ = list(itertools.chain.from_iterable(y))
    counter = len(y_)

    assert counter == len(y_hat)

    for i in range(0, counter, chunk_size):
        chunk_y = y_[i:i + chunk_size]
        chunk_y_hat = y_hat[i:i + chunk_size]
        pairs = zip(chunk_y, chunk_y_hat)

        results = Parallel(n_jobs=int(cpu_count()) - 1)(
            delayed(mapping_labels)(pair, label_mapping) for pair in pairs)
        yield pd.DataFrame(results)
