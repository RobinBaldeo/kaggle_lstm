import pandas as pd
import os, sys, json, pdb

from data_processing.data_layer import DataParsing
from utils.misc import flatten_out_list
from sklearn.model_selection import train_test_split

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dp = DataParsing(chunk_size=400, overlap=2)

    df = (pd.read_json('data/test.json')
          .head(100)
          .reset_index()
          )

    unique_tokens = flatten_out_list(df.tokens.to_list())
    unique_labels = flatten_out_list(df.labels.to_list())

    x_train, x_val = train_test_split(df, test_size=0.2, random_state=42)

    x_train = dp.fit_transform(x_train)

    x_val = dp.fit_transform(x_val)



    pdb.set_trace()
