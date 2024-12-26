import shap
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

# データセットの読み込み
boston = load_boston()
X, y = boston.data, boston.target

# データセットの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの訓練
model = xgboost.XGBRegressor()
model.fit(X_train, y_train)

# SHAP値の計算
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# SHAP値のプロット
shap.summary_plot(shap_values, X_test, feature_names=boston.feature_names)