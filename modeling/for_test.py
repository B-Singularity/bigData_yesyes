import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import shap
import matplotlib.pyplot as plt

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
file_path = "../preprocessing/investing/merged_data_with_gdp (1).csv"
df = pd.read_csv(file_path)

# -----------------------------
# 2. 타깃 생성 (분위수 기준)
# -----------------------------
q85 = df['gdp_통신업'].quantile(0.85)
q60 = df['gdp_통신업'].quantile(0.60)
q30 = df['gdp_통신업'].quantile(0.30)

def assign_tier(x):
    if x >= q85:
        return "S"
    elif x >= q60:
        return "A"
    elif x >= q30:
        return "B"
    else:
        return "C"

df['talent_tier'] = df['gdp_통신업'].apply(assign_tier)

# -----------------------------
# 3. Feature / Target 분리
# -----------------------------
X = df.drop(columns=['Unnamed: 0', 'talent_tier', 'gdp_통신업'])
y = df['talent_tier']
X = X.fillna(0)

# -----------------------------
# 4. 시계열 CV 설정
# -----------------------------
n_splits = 5
window_size = int(len(df) / (n_splits + 1))  # 학습 데이터 길이

accuracy_list = []
roc_auc_list = []
feature_importances = np.zeros(X.shape[1])

# LabelBinarizer: ROC-AUC 계산용
lb = LabelBinarizer()
lb.fit(y)  # 전체 데이터 기준

for i in range(n_splits):
    train_end = window_size * (i + 1)
    test_start = train_end
    test_end = train_end + window_size

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[test_start:test_end]
    y_test = y.iloc[test_start:test_end]

    # -----------------------------
    # 5. 모델 학습
    # -----------------------------
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # -----------------------------
    # 6. 예측
    # -----------------------------
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    # -----------------------------
    # 7. 평가
    # -----------------------------
    acc = accuracy_score(y_test, y_pred)
    accuracy_list.append(acc)

    # ROC-AUC 계산 시 클래스 결손 처리
    y_test_bin = lb.transform(y_test)
    if y_test_bin.shape[1] == y_pred_proba.shape[1]:
        roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class="ovr")
    else:
        roc_auc = np.nan
    roc_auc_list.append(roc_auc)

    # 특성 중요도 누적
    feature_importances += clf.feature_importances_

    print(f"Fold {i+1}: Accuracy={acc:.4f}, ROC-AUC={roc_auc if not np.isnan(roc_auc) else 'nan'}")

# -----------------------------
# 8. 평균 결과
# -----------------------------
print("\n평균 Accuracy: %.4f (+/- %.4f)" % (np.mean(accuracy_list), np.std(accuracy_list)))
print("평균 ROC-AUC: %.4f (+/- %.4f)" % (np.nanmean(roc_auc_list), np.nanstd(roc_auc_list)))

# -----------------------------
# 9. Feature Importance 시각화
# -----------------------------
feature_importances /= n_splits
fi_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(fi_df['feature'], fi_df['importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance (Average over folds)")
plt.show()

# -----------------------------
# 10. SHAP 시각화 (마지막 fold 기준)
# -----------------------------
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
