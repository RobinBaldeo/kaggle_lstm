import pandas as pd
import os, sys, json, pdb

from data_processing.data_layer import  chunk_data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = (pd.read_json('data/test.json')
          .head(100)
          )

    print(chunk_data(df))




