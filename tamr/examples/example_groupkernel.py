import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import shap

# データセットの読み込み
california = fetch_california_housing()
X, y = california.data, california.target

# 特徴量の名前の設定
print(california.feature_names)

# 特徴量のグループ化（例：2つずつの特徴量を1グループ）
# california：'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude’
#groups = [[0],[1],[2],[3],[4],[5],[6],[7]]
#groups = [[0,1],[2,3],[4,5],[6,7]]
groups = [[1],[2],[3], [4],[5],[0,6,7]]

# グループ化された特徴量名
original_feature_names = california.feature_names
group_feature_names = [
    "_".join([original_feature_names[i] for i in group]) for group in groups
]
print("Grouped feature names:", group_feature_names)

# データセットの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの訓練
model = xgboost.XGBRegressor()
model.fit(X_train, y_train)

# 背景データの作成
background = shap.sample(X_train, 100)

# カスタムエクスプレイナーの初期化
class GroupKernelExplainer(shap.KernelExplainer):
    def __init__(self, model, data, groups=None, **kwargs):
        self.groups = groups
        super().__init__(model, data, **kwargs)

    def varying_groups(self, x):
        varying = np.zeros(len(self.groups))
        for i, group in enumerate(self.groups):
            for feature in group:
                if not np.isclose(x[0, feature], self.data.data[:, feature]).all():
                    varying[i] = 1
                    break
        return np.where(varying == 1)[0]

explainer = GroupKernelExplainer(model.predict, background, groups=groups)

# SHAP値の計算
shap_values = explainer.shap_values(X_test)

# グループ数
num_groups = len(group_feature_names)

# SHAP値とテストデータをグループ数に基づいてスライス
shap_values_filtered = shap_values[:, :num_groups]
X_test_filtered = X_test[:, :num_groups]

# SHAP値のプロット
"""
"""
# 1. Summary Plot
shap.summary_plot(shap_values_filtered, X_test_filtered, feature_names=group_feature_names)
"""
# 2. Bar Plot
shap.summary_plot(shap_values_filtered, X_test_filtered, feature_names=group_feature_names, plot_type="bar")
# 3. Force Plot
## データポイントのインデックス（例: 0番目）
index = 0
# フォースプロット
shap.force_plot(
    explainer.expected_value,
    shap_values_filtered[index],
    X_test_filtered[index],
    feature_names=group_feature_names
)
# 4. Waterfall Plot
# データポイントのインデックス（例: 0番目）
index = 0
# ウォーターフォールプロット
shap.plots.waterfall(shap.Explanation(
    values=shap_values_filtered[index],
    base_values=explainer.expected_value,
    data=X_test_filtered[index],
    feature_names=group_feature_names
))
# 5. Dependence Plot
# グループインデックス（例: 0番目のグループ）
group_index = 0
# 依存関係プロット
shap.dependence_plot(group_index, shap_values_filtered, X_test_filtered, feature_names=group_feature_names)
# 6. Heatmap Plot
shap.plots.heatmap(shap.Explanation(
    values=shap_values_filtered,
    base_values=explainer.expected_value,
    data=X_test_filtered,
    feature_names=group_feature_names
))
"""
