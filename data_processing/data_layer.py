import pdb

import pandas as pd
import spacy
from joblib import Parallel, delayed
from collections import namedtuple
import itertools, pdb

nlp = spacy.load("en_core_web_sm")

chunk_holder = namedtuple('chunk_holder', 'doc_id position tokens labels pos')


def pos_tags(r):
    doc, to_tokens = r
    return [token.tag_ for token in doc if token.text in to_tokens]


def chunk_sequence(r, chunk_size=400, overlap=2):
    lst = []

    doc, tokens, labels, pos = r
    for i in range(0, len(tokens), chunk_size - overlap):
        chunks_token = tokens[i:i + chunk_size]
        if isinstance(labels, list):
            chunks_label = labels[i:i + chunk_size]
        else:
            chunks_label = 0

        chunks_pos = pos[i:i + chunk_size]
        position = [i for i in range(i, i + len(chunks_token))]
        lst.append(chunk_holder(doc_id=doc
                                , position=position
                                , tokens=chunks_token
                                , labels=chunks_label
                                , pos=chunks_pos
                                )
                   )
    return lst


def chunk_data(df: pd.DataFrame):
    df_ = (
        df
        .copy()
        .assign(
            docs=lambda f: list(nlp.pipe(f.full_text)),
            pos=lambda f: f.loc[:, ['docs', 'tokens']].apply(pos_tags, axis=1),
        )
    )

    labels_lst = df.get('labels', [0] * df.shape[0])

    pairs = zip(df_.document.to_list(), df_.tokens.to_list(), labels_lst, df_.pos.to_list())
    results = Parallel(n_jobs=-1)(delayed(chunk_sequence)(pair) for pair in pairs)
    df2 = pd.DataFrame(itertools.chain.from_iterable(results))

    if 'labels' not in df.columns:
        df2.drop(columns='labels', inplace=True)

    return df2
