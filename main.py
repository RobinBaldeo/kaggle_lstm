import pandas as pd
import os, sys, json, pdb

from data_processing.data_layer import  DataParsing

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dp = DataParsing()

    df = (pd.read_json('data/test.json')
          .head(100)
          )

    print(dp.fit_transform(df))




