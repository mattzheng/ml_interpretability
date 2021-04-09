
# train an XGBoost model
import xgboost
import shap
import pandas as pd

# 获取数据
X, y = shap.datasets.boston()

# train an XGBoost model
model = xgboost.XGBRegressor().fit(X, y)

# 计算概率值
probas_xgb = pd.Series(model.predict(X),index=X.index)
probas_xgb

# 获得全样本的shap值
explainer = shap.Explainer(model)
shap_values = explainer(X)

shap_df = pd.DataFrame([list(shap_values[n].values)  for n in range(X.shape[0])],columns = X.columns )
shap_df

# shap值 映射向 概率
from util import shap2deltaprob_v2
shap_temp = shap2deltaprob_v2(shap_df, 
                   probas_xgb,
                   func_shap2probas = 'interp1d')
shap_temp


# 画出重点shap值
shap_temp = shap_temp.head().applymap(lambda x:('+'if x>0 else '')+str(round(x*100,2))+'%')

shap_temp.style.apply(lambda x: ["background:orangered" if float(v[:-1])<0 else "background:lightgreen"
                                for v in x], axis = 1)