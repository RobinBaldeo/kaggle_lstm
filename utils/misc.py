

import functools
import pdb
import time, math, itertools
import numpy as np


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







