import configparser

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

class HiringModel:

    def __init__(self, config):
        self.config = config
        self.data = None
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.features = self.config['features']
        self.target = self.config['target']

    def prepare_modeling_data(self):
        quantiles = self.config['tier_quantiles']
        clean_growth = self.data['growth_rate_qoq'].dropna()
        s_tier = clean_growth.quantile(quantiles['S'])
        a_tier = clean_growth.quantile(quantiles['A'])
        b_tier = clean_growth.quantile(quantiles['B'])

        def assign_talent_tier(rate):
            if rate >= s_tier: return 'S'
            elif rate >= a_tier: return 'A'
            elif rate >= b_tier: return 'B'
            else: return 'C'

        self.data['talent_tier'] = self.data['growth_rate_qoq'].apply(assign_talent_tier)

        model_df = self.data[self.features + [self.target]].dropna()
        model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

        X = model_df[self.features]
        Y = model_df[self.target]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.Y_train = Y_train
        self.Y_test = Y_test

    def train(self):
        if self.X_train_scaled is None:
            raise ValueError("모델링 데이터를 먼저 준비해야 합니다.")

        self.model = LogisticRegression(
            multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42
        )
        self.model.fit(self.X_train_scaled, self.Y_train)

    def evaluate(self):
        if self.model is None:
            raise ValueError("모델을 먼저 학습시켜야 합니다. train()을 실행하세요.")

        accuracy = self.model.score(self.X_test_scaled, self.Y_test)
        return accuracy

    def get_coefficients(self):
        if self.model is None:
            raise ValueError("모델을 먼저 학습시켜야 합니다. train()을 실행하세요.")

        coeffs = pd.DataFrame(
            self.model.coef_.T,
            index=self.features,
            columns=self.model.classes_
        )
        desired_order = ['S', 'A', 'B', 'C']
        if all(c in coeffs.columns for c in desired_order):
            coeffs = coeffs[desired_order]
        return coeffs

    def predict_new_data(self, data_dict):
        if self.model is None or self.scaler is None:
            raise ValueError("모델을 먼저 학습시켜야 합니다. train()을 실행하세요.")

        new_df = pd.DataFrame([data_dict])
        new_df = new_df[self.features]
        new_df_scaled = self.scaler.transform(new_df)

        probabilities = self.model.predict(new_df_scaled)

        return pd.DataFrame(probabilities, columns=self.model.classes_, index=['Predicted Probability'])


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
