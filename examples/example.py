import shap
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# データセットの読み込み
california = fetch_california_housing()
X, y = california.data, california.target

# データセットの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの訓練
model = xgboost.XGBRegressor()
model.fit(X_train, y_train)

# SHAP値の計算
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

print(model)

# shap_values have [.values, .base_values, .data]
# .values = array([[-0.47621608,  0.00522793, -0.11823769, ..., -0.37759832],
#                  [-0.5051474 ,  0.02292835, -0.11505361, ...,-0.22896677],
#                   ...])
# .base_values = array([2.0718894, 2.0718894, 2.0718894, ..., 2.0718894], dtype=float32)
# .data = array([[1.6812, 25., 4.19220056, ..., -119.01],
#                [2.5313, 30., 5.03938356, ..., -119.46],
#                ...])

"""
# SHAP値のプロット
shap.summary_plot(shap_values, X_test, feature_names=california.feature_names)
"""