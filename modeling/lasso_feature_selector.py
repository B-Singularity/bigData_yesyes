import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np

class LassoFeatureSelector:

    def __init__(self, clean_dataframe, config):
        self.data = clean_dataframe
        self.config = config
        self.features = self.config['features']
        self.target = self.config['target']
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.scaler = None
        self.model = None

    def prepare_modeling_data(self):
        model_df = self.data[self.features + [self.target]].dropna()
        model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

        X = model_df[self.features]
        Y = model_df[self.target]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.Y_train = Y_train
        self.Y_test = Y_test

    def train(self):
        if self.X_train_scaled is None:
            raise ValueError("모델링 데이터를 먼저 준비해야 합니다.")

        self.model = LassoCV(cv=5, random_state=42, max_iter=10000)
        self.model.fit(self.X_train_scaled, self.Y_train)

    def evaluate(self):
        if self.model is None:
            raise ValueError("모델을 먼저 학습시켜야 합니다. train()을 실행하세요.")

        r_squared = self.model.score(self.X_test_scaled, self.Y_test)
        return r_squared

    def get_coefficients(self):
        if self.model is None:
            raise ValueError("모델을 먼저 학습시켜야 합니다.")

        coeffs = pd.DataFrame(
            self.model.coef_,
            index=self.features,
            columns=['Coefficient']
        )
        return coeffs



