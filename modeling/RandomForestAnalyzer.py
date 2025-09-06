import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np


# --- 1. 랜덤 포레스트 모델 클래스 정의 ---
class RandomForestAnalyzer:
    """
    전처리된 데이터프레임을 입력받아 랜덤 포레스트 회귀 모델을 실행하고 평가하는 클래스.
    """

    def __init__(self, clean_dataframe, config):
        self.data = clean_dataframe
        self.config = config
        self.model = None
        self.features = self.config['features']
        self.target = self.config['target']
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_modeling_data(self):
        """모델링에 사용할 데이터를 준비하고, 학습/테스트용으로 분리합니다."""
        # Date 컬럼이 있다면 datetime으로 변환
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])

        # NaN, inf 값들을 가진 행들을 제거하여 데이터 정제
        model_df = self.data[self.features + [self.target]].dropna()
        model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

        X = model_df[self.features]
        y = model_df[self.target]

        # 데이터를 훈련용과 테스트용으로 분리
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print("모델링 데이터 준비 완료.")

    def train(self):
        """랜덤 포레스트 모델을 학습시킵니다."""
        # n_estimators는 생성할 의사결정나무의 개수, n_jobs=-1은 모든 CPU 코어를 사용하라는 의미입니다.
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)
        print("모델 학습 완료.")

    def evaluate(self):
        """학습된 모델의 성능(R-squared)을 평가합니다."""
        y_pred = self.model.predict(self.X_test)
        return r2_score(self.y_test, y_pred)

    def get_feature_importances(self):
        """
        각 변수가 예측에 얼마나 중요한지를 반환합니다.
        """
        importances = self.model.feature_importances_
        # 변수 중요도를 보기 쉽게 데이터프레임으로 만듭니다.
        return pd.DataFrame(
            importances,
            index=self.features,
            columns=['Importance']
        ).sort_values(by='Importance', ascending=False)


# --- 2. 메인 실행 블록 ---
if __name__ == '__main__':
    try:
        # 1. 대표님이 올려주신 최종 데이터를 불러옵니다.
        data_filepath = "../preprocessing/final_analytical_data_featured.csv"
        clean_data = pd.read_csv(data_filepath)

        print(f"'{data_filepath}' 파일을 성공적으로 불러왔습니다.")

        # 2. 랜덤 포레스트 분석을 위한 설정을 정의합니다.
        config_rf = {
            'target': 'growth_rate_qoq',
            'features': [
                'unemployment_rate',
                'BSI_Composite',
                'real_wage_growth',
                'population',
                'productivity_index',
                'unemployment_rate_MA3',
                'BSI_Composite_MA3',
                'unemployment_rate_change3',
                'BSI_Composite_change3'
            ]
        }

        # 3. 모델 클래스를 실행합니다.
        analyzer = RandomForestAnalyzer(clean_data, config_rf)
        analyzer.prepare_modeling_data()
        analyzer.train()

        # 4. 결과를 확인합니다.
        r2_score_result = analyzer.evaluate()
        feature_importances = analyzer.get_feature_importances()

        print("\n" + "=" * 40)
        print("     랜덤 포레스트 분석 최종 결과")
        print("=" * 40)
        print(f"\n## 모델 평가 결과 (R-squared) ##\n{r2_score_result:.4f}")
        print("\n## 변수별 영향력 (Feature Importance) ##")
        print(feature_importances)

    except FileNotFoundError:
        print(f"\n[오류] 파일 없음: '{data_filepath}' 파일을 찾을 수 없습니다.")
    except KeyError as e:
        print(f"\n[오류] 컬럼 없음: {e} 컬럼이 파일에 존재하지 않습니다. 확인해주세요.")
    except Exception as e:
        print(f"\n[오류] 분석 중 예외가 발생했습니다: {e}")