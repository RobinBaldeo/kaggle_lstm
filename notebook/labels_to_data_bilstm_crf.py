import pdb

import pandas as pd
import itertools
from collections import namedtuple
from typing import Dict
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from misc import process_time

prediction_elements = namedtuple('prediction_elements', 'doc_id position tokens predictions')


def convert_doc_id_lst(row):
    doc_id, position, *_ = row
    return [doc_id] * len(position)


def created_output_df(pairs, mapping):
    doc_id, position, token, predictions = pairs
    predict_mapped = mapping.get(predictions)
    to_ret = prediction_elements(doc_id=doc_id, position=position, tokens=token, predictions=predict_mapped)

    return to_ret


def process_in_chunks(chunk_size, doc_id, position, tokens, predictions_lst, reverse_mapping):
    doc_id_lst, position_lst, tokens_lst = map(list, [doc_id, position, tokens])
    counter = len(doc_id_lst)
    assert counter == len(predictions_lst)

    # pdb.set_trace()
    for i in range(0, counter, chunk_size):
        chunk_doc_id = doc_id_lst[i:i + chunk_size]
        chunk_position = position_lst[i:i + chunk_size]
        chunk_tokens = tokens_lst[i:i + chunk_size]
        chunk_predictions = predictions_lst[i:i + chunk_size]

        pairs = zip(chunk_doc_id, chunk_position, chunk_tokens, chunk_predictions)
        results = Parallel(n_jobs=int(cpu_count()) - 1)(
            delayed(created_output_df)(pair, reverse_mapping) for pair in pairs)
        yield pd.DataFrame(results)


@process_time
def convert_to_labels(df, prediction, mapping: Dict, chunk_size):
    """
    TODO
    """

    reverse_mapping = {j: i for i, j in mapping.items()}
    new_df = (
        df
        .copy()
        .assign(
            expand_dic_id=lambda x: x.apply(convert_doc_id_lst, axis=1),
        )
    )
    # pdb.set_trace()

    flat_doc_id = itertools.chain.from_iterable(new_df.expand_dic_id.to_list())
    flat_position = itertools.chain.from_iterable(new_df.position.to_list())
    flat_tokens = itertools.chain.from_iterable(new_df.tokens.to_list())
    flat_predictions = itertools.chain.from_iterable(itertools.chain.from_iterable(prediction))  # Nested List

    no_padding = [item for item in flat_predictions if item != mapping["<PAD>"]]

    results_df = pd.concat(process_in_chunks(
        chunk_size=chunk_size,
        doc_id=flat_doc_id,
        position=flat_position,
        tokens=flat_tokens,
        predictions_lst=no_padding,
        reverse_mapping=reverse_mapping)
    ).reset_index(drop=True)

    return (
        pd.merge(
            (results_df
             .drop(columns='tokens')
             .groupby(['doc_id', 'position', 'predictions'])
             .agg(
                count_=('predictions', 'count')
            )
             .reset_index()
             .sort_values(by='count_', ascending=False)
             .drop_duplicates(subset=['doc_id', 'position'], keep='first')
             .reset_index(drop=True)
             .drop(columns='count_')
             ),
            (
                results_df
                .drop(columns='predictions')
                .drop_duplicates(subset=['doc_id', 'position'], keep='first')
            )
            , right_on=['doc_id', 'position']
            , left_on=['doc_id', 'position']

        )
        .rename(columns={
            'doc_id': 'document',
            'tokens': 'words',
            'position': 'tokens',
            'predictions': 'label'
        })
        .sort_values(by=['document', 'tokens'])
        .loc[:, ['document', 'tokens', 'label', 'words']]
        .reset_index(drop=True)
    )

