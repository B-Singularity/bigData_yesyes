import pandas as pd
import numpy as np

class RateOfChangeFeatureGenerator:
    def __init__(self, target_column, periods):
        self.target_column = target_column
        self.periods = periods

    def transform(self, df):
        df_new = df.copy()
        for period in self.periods:
            col_name = f'{self.target_column}_roc_{period}'
            df_new[col_name] = df_new[col_name].rolling(period).mean()
            df_new[col_name] = df_new[self.target_column].pct_change(periods=period) * 100
        return df_new


