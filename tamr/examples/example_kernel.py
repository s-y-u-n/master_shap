import sys
#sys.path.insert(0, '/Users/tamurashuntarou/CML/code/master_shap')
sys.path.insert(0, '/Users/tamr/CMIS/code/master_shap')
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import shap

# データセットの読み込み（20サンプルのみ）
california = fetch_california_housing()
X, y = california.data[:20], california.target[:20]

# 30サンプルだけ抽出
X, y = X[:30], y[:30]

# データセットの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(X_train.shape, X_test.shape)

# KernelExplainer.__init__  KernelExplainer.__call__    KernelExplainer.shap_values を回す
# X_testの分だけ    KernelExplainer.explain   KernelExplainer.run を回す

# モデルの訓練
model = xgboost.XGBRegressor()
model.fit(X_train, y_train)

# SHAP値の計算
background = shap.sample(X_train, 5)
explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer(X_test)
