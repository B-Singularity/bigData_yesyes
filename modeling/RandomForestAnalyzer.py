import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, classification_report, confusion_matrix
import numpy as np
import optuna
from imblearn.over_sampling import SMOTE


class RandomForestAnalyzer:
    """
    전처리된 데이터프레임을 입력받아 랜덤 포레스트 회귀 또는 분류 모델을 실행하고 평가하는 클래스.
    """

    def __init__(self, clean_dataframe, config):
        self.data = clean_dataframe
        self.config = config
        # --- 수정: 'regression' -> 'regressor' 오타 수정 ---
        self.model_type = self.config.get('model_type', 'regressor')
        self.model = None
        self.features = self.config['features']
        self.target = self.config['target']
        # --- 수정: 변수명을 표준 컨벤션에 맞게 소문자로 통일 ---
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.best_params = None
        self.X, self.y = [None] * 2

    def prepare_modeling_data(self):
        if self.model_type == 'classifier':
            print("분류 모드: Y값 (talent_tier)을 생성합니다...")
            quantiles = self.config['tier_quantiles']
            clean_growth = self.data['growth_rate_qoq'].dropna()
            s_tier = clean_growth.quantile(quantiles['S'])
            a_tier = clean_growth.quantile(quantiles['A'])
            b_tier = clean_growth.quantile(quantiles['B'])

            def assign_talent_tier(rate):
                if rate >= s_tier:
                    return 'S'
                elif rate >= a_tier:
                    return 'A'
                elif rate >= b_tier:
                    return 'B'
                else:
                    return 'C'

            self.data[self.target] = self.data['growth_rate_qoq'].apply(assign_talent_tier)

        model_df = self.data[self.features + [self.target]].dropna()
        model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

        self.X = model_df[self.features]
        self.y = model_df[self.target]

        if self.model_type == 'classifier':
            smote = SMOTE(random_state=42)
            self.X, self.y = smote.fit_resample(self.X, self.y)
            print("SMOTE 적용: 클래스 불균형 해결 완료.")

        stratify_option = self.y if self.model_type == 'classifier' else None
        # --- 수정: 변수명을 소문자로 통일 ---
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=stratify_option
        )
        print("모델링 데이터 준비 완료.")

    def hyperparameter_tuning(self, n_trials=200):
        print("\n하이퍼파라미터 튜닝을 시작합니다...")

        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'class_weight': 'balanced' if self.model_type == 'classifier' else None
            }
            if self.model_type == 'regressor':
                model = RandomForestRegressor(**param, random_state=42, n_jobs=-1)
                model.fit(self.X_train, self.y_train)
                return r2_score(self.y_test, model.predict(self.X_test))
            else:
                model = RandomForestClassifier(**param, random_state=42, n_jobs=-1)
                model.fit(self.X_train, self.y_train)
                return accuracy_score(self.y_test, model.predict(self.X_test))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        print("튜닝 완료.")
        print(f"최고 점수 (Best Score): {study.best_value:.4f}")
        print(f"최적 하이퍼파라미터: {self.best_params}")

    def cross_validation_score(self, cv_folds=5):
        print(f"\n{cv_folds}-fold 교차검증을 실행합니다...")
        params = {'random_state': 42, 'n_jobs': -1}
        if self.best_params:
            params.update(self.best_params)
        else:
            params['n_estimators'] = 100

        if self.model_type == 'regressor':
            model = RandomForestRegressor(**params)
            cv_scores = cross_val_score(model, self.X, self.y, cv=cv_folds, scoring='r2')
            metric_name = "R2 Score"
        else:
            model = RandomForestClassifier(**params)
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, self.X, self.y, cv=skf, scoring='accuracy')
            metric_name = "Accuracy"

        print(f"\n== 교차검증 결과 ({metric_name}) ==")
        print(f"각 fold 점수: {cv_scores}")
        print(f"평균: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        return cv_scores

    def train(self, use_best_params=False):
        params = {'random_state': 42, 'n_jobs': -1}
        if use_best_params and self.best_params:
            print("\n최적 하이퍼파라미터로 최종 모델을 학습합니다...")
            params.update(self.best_params)
        else:
            print("\n기본 하이퍼파라미터로 모델을 학습합니다...")
            params['n_estimators'] = 100

        if self.model_type == 'regressor':
            self.model = RandomForestRegressor(**params)
        elif self.model_type == 'classifier':
            self.model = RandomForestClassifier(**params)

        self.model.fit(self.X_train, self.y_train)
        print("모델 학습 완료.")

    def evaluate(self, detailed=False):
        y_pred = self.model.predict(self.X_test)
        if self.model_type == 'regressor':
            return r2_score(self.y_test, y_pred)
        elif self.model_type == 'classifier':
            accuracy = accuracy_score(self.y_test, y_pred)
            if detailed:
                print(f"\n== 상세 분류 성능 평가 ==")
                print(f"정확도: {accuracy:.4f}")
                print("\n분류 리포트:")
                print(classification_report(self.y_test, y_pred, zero_division=0))
                print("\n혼동 행렬:")
                print(confusion_matrix(self.y_test, y_pred))
            return accuracy

    # --- 수정: 중복된 메서드 제거 ---
    def get_feature_importances(self):
        importances = self.model.feature_importances_
        return pd.DataFrame(
            importances, index=self.features, columns=['Importance']
        ).sort_values(by='Importance', ascending=False)

    def predict_probabilities(self, data_dict):
        if self.model_type != 'classifier':
            return "확률 예측은 'classifier' 모드에서만 가능합니다."
        new_df = pd.DataFrame([data_dict])[self.features]
        probabilities = self.model.predict_proba(new_df)
        return pd.DataFrame(probabilities, columns=self.model.classes_, index=['Predicted Probability'])

if __name__ == '__main__':
    try:
        data_filepath = "../preprocessing/hiring/final_analytical_data_advanced_featured.csv"
        clean_data = pd.read_csv(data_filepath)
        print(f"'{data_filepath}' 파일을 성공적으로 불러왔습니다.\n")

        # --- 분석 모드 선택 ---
        CHOSEN_MODE = 'classifier'

        if CHOSEN_MODE == 'classifier':
            config = {
                'model_type': 'classifier',
                'target': 'talent_tier',
                'tier_quantiles': {'S': 0.85, 'A': 0.60, 'B': 0.30},
                'features': [
                    # 기본 변수
                    'unemployment_rate',
                    'BSI_Composite',
                    'real_wage_growth',
                    'growth_rate_qoq_lag1',
                    'population',
                    'GFCF_ICT_Real',  # GFCF 변수 추가

                    # 단기 추세 변수
                    'unemployment_rate_MA3',
                    'BSI_Composite_MA3',
                    'unemployment_rate_change3',
                    'BSI_Composite_change3',

                    # 고급 특성 변수 (중장기 추세 및 변동성)
                    'unemployment_rate_MA6',
                    'BSI_Composite_MA12',
                    'real_wage_growth_change12',
                    'unemployment_rate_std3',
                    'BSI_Composite_std6'
                ]
            }
            analyzer = RandomForestAnalyzer(clean_data, config)
            analyzer.prepare_modeling_data()

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
    except Exception as e:
        print(f"\n[오류] 분석 중 예외가 발생했습니다: {e}")