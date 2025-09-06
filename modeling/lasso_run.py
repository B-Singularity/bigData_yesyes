from modeling.lasso_feature_selector import LassoFeatureSelector
import pandas as pd

if __name__ == '__main__':
    data_filepath = "../preprocessing/final_analytical_data.csv"
    clean_analytical_data = pd.read_csv(data_filepath)


    # 2. 라쏘 회귀 분석을 위한 설정 정의
    config_lasso = {
        # Y값: '인재의 질'을 나타내는 노동생산성 증가율로 설정
        'target': 'growth_rate_qoq',
        # X값: Y값을 예측하는데 사용할 모든 후보 변수들
        'features': ['unemployment_rate', 'BSI', 'real_wage_growth', 'population']
    }

    # 3. 깨끗한 데이터를 모델 클래스에 주입하여 분석 실행
    selector = LassoFeatureSelector(clean_analytical_data, config_lasso)
    selector.prepare_modeling_data()
    selector.train()

    # 4. 결과 확인
    r2_score = selector.evaluate()
    coefficients = selector.get_coefficients()
    print(f"\n## 모델 평가 결과 (R-squared) ##\n{r2_score:.4f}")
    print("\n## 변수별 영향력 (회귀 계수) ##")
    print(coefficients)