import numpy as np
import pandas as pd

class MovingAverageFeatureGenerator:
    def __init__(self, target_column, windows):
        self.target_column = target_column
        self.windows = windows

    def transform(self, df):
        df_new = df.copy()
        df_new = df.copy()
        for window in self.windows:
            col_name = f'{self.target_column}_ma_{window}'
            df_new[col_name] = df_new[self.target_column].rolling(window=window).mean()
        return df_new