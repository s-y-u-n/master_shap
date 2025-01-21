import sys
#sys.path.append('/Users/tamurashuntarou/CML/code/master_shap')
sys.path.append('/Users/tamr/CMIS/code/master_shap/shap')
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
background = shap.sample(X_train, 3)
explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer(X_test)
