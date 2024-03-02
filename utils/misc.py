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


@process_time
def f5_score_mapping(df_y, df_y_hat, label_mapping: dict):

    raw_df_y = (
        df_y
        .copy()
        .loc[:, ['doc_id', 'position', 'labels']]
        .assign(
            lst_doc_id=lambda x: x.apply(lambda g: [g.doc_id] * len(g.position), axis=1),
        )
        .drop(columns='doc_id')
    )

    doc_id_lst, position_lst, labels_lst = map(lambda a: list(itertools.chain.from_iterable(a)),
                                               [raw_df_y.lst_doc_id,
                                                raw_df_y.position,
                                                raw_df_y.labels]
                                               )

    return (
        pd.merge(
            (pd.DataFrame({
                'doc_id': doc_id_lst,
                'position': position_lst,
                'labels': labels_lst
            })
             .query('labels !="O"')
             .drop_duplicates(subset=['doc_id', 'position'])
             .rename(columns={'labels': 'y_'})
             .reset_index(drop=True)
             .set_index(['doc_id', 'position'])
             ),
            (
                df_y_hat
                .loc[:, ['document', 'tokens', 'label']]
                .rename(columns={
                    'document': 'doc_id',
                    'tokens': 'position',
                    'label': 'y_hat'
                }
                )
                .query('labels !="O"')
                .set_index(['doc_id', 'position'])
            )
            , how='outer'
            , right_index=True
            , left_index=True
        )
        .fillna('O')
        .assign(
            y_idx=lambda x: x.y_.map(lambda p: label_mapping.get(p, -1)),
            y_hat_idx= lambda x: x.y_hat.map( lambda p: label_mapping.get(p, -1))
        )
    )
