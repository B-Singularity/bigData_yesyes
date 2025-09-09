import RandomForestAnalyzer
import pandas as pd


if __name__ == '__main__':
    try:
        data_filepath = "../preprocessing/project/simulation_data_with_features.csv"
        clean_data = pd.read_csv(data_filepath, index_col='date', parse_dates=True)
        features = clean_data.columns.tolist()
        print(f"'{data_filepath}' 파일을 성공적으로 불러왔습니다.\n")

        # --- 분석 모드 선택 ---
        CHOSEN_MODE = 'classifier'

        if CHOSEN_MODE == 'classifier':
            features = clean_data.columns.tolist()
            config = {
                'model_type': 'classifier',
                'target': 'talent_tier',
                'tier_quantiles': {'S': 0.85, 'A': 0.60, 'B': 0.30},
                'features': [col for col in features if col not in ['BSI_Composite', 'talent_tier']]
            }
            analyzer = RandomForestAnalyzer.RandomForestAnalyzer(clean_data, config)
            analyzer.prepare_modeling_data('BSI_Composite')

            # --- 여기가 핵심 실행 흐름 ---
            # 1. 하이퍼파라미터 튜닝 실행
            analyzer.hyperparameter_tuning(n_trials=50)

            # 2. 튜닝으로 찾은 최적의 파라미터로 최종 모델 학습
            analyzer.train(use_best_params=True)

            # 3. 최종 모델 평가 및 결과 확인
            score = analyzer.evaluate()
            importances = analyzer.get_feature_importances()

            print("\n" + "=" * 50)
            print("     랜덤 포레스트 [분류] 튜닝 후 최종 결과")
            print("=" * 50)
            print(f"\n## 모델 평가 결과 (튜닝 후 정확도) ##\n{score:.4f}")
            print("\n## 변수별 영향력 (Feature Importance) ##")
            print(importances)

        # (회귀 모드 실행 부분은 생략)

    except FileNotFoundError:
        print(f"\n[오류] 파일 없음: '{data_filepath}' 파일을 찾을 수 없습니다.")
