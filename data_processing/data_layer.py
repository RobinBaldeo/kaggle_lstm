import pdb

import pandas as pd
import spacy
from joblib import Parallel, delayed
from collections import namedtuple
import itertools, pdb
from functools import partial
from utils.misc import process_time


class DataParsing:

    def __init__(self, chunk_size=400, overlap=2):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunk_holder = namedtuple('chunk_holder',
                                       ['doc_id', 'position', 'tokens', 'labels', 'pos']
                                       )

    def _pos_tags(self, r):
        """
        verify that the tokens are exactly the same as kaggle token
        :param pandas column:
        :return:
        """
        doc, to_tokens = r
        return [token.tag_ for token in doc if token.text in to_tokens]

    def _chunk_sequence(self, r):
        """
        the method takes does the chunking of the txt into n chunks with overlaps
        :param r: each row in the dataframe
        :return: list of chunks for labels, pos, and tokens
        """
        lst = []

        doc, tokens, labels, pos, chunk_size, overlap = r
        for i in range(0, len(tokens), chunk_size - overlap):
            chunks_token = tokens[i:i + chunk_size]
            if isinstance(labels, list):
                chunks_label = labels[i:i + chunk_size]
            else:
                chunks_label = 0

            chunks_pos = pos[i:i + chunk_size]
            position = [i for i in range(i, i + len(chunks_token))]
            lst.append(self.chunk_holder(doc_id=doc
                                         , position=position
                                         , tokens=chunks_token
                                         , labels=chunks_label
                                         , pos=chunks_pos
                                         )
                       )
        return lst

    @process_time
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        preform the chunking of the json file for further processing
        :param df:
        :return:
        """
        nlp = spacy.load("en_core_web_sm")
        df_ = (
            df
            .copy()
            .assign(
                docs=lambda f: list(nlp.pipe(f.full_text)),
                pos=lambda f: f.loc[:, ['docs', 'tokens']].apply(self._pos_tags, axis=1),
            )
        )
        # pdb.set_trace()

        labels_lst = df.get('labels', [0] * df.shape[0])
        async_chunk_size = [self.chunk_size] * df.shape[0]
        async_overlap = [self.overlap] * df.shape[0]

        pairs = zip(df_.document.to_list(),
                    df_.tokens.to_list(),
                    labels_lst,
                    df_.pos.to_list(),
                    async_chunk_size,
                    async_overlap
                    )
        results = Parallel(n_jobs=-1)(delayed(self._chunk_sequence)(pair) for pair in pairs)
        df2 = pd.DataFrame(itertools.chain.from_iterable(results))

        if 'labels' not in df.columns:
            df2.drop(columns='labels', inplace=True)

        return df2
