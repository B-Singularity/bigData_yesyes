import pandas as pd
import numpy as np

class LagFeatureGenerator:
    def __init__(self, target_column, lags):
        self.target_column = target_column
        self.lags = lags

    def transform(self, df):
        df_new = df.copy()
        for lag in self.lags:
            col_name = f'{self.target_column}_lag_{lag}'
            df_new[col_name] = df_new[self.target_column].shift(lag)
        return df_new

