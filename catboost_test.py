

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
#import shap
from scipy.interpolate import interp1d
#shap.initjs()

# 加载数据
train_data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')

X_all=pd.concat([train_data[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']],
                 test_data[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
                ]).set_index('PassengerId')
y_train=train_data[['PassengerId','Survived']].set_index('PassengerId')['Survived']

num_columns=['Age','SibSp','Parch','Fare']
cat_columns=list(filter(lambda x:x not in num_columns,X_all.columns))

# categorical features: finn nan's with string
X_all[cat_columns]=X_all[cat_columns].astype(str)
X_all[cat_columns]=X_all[cat_columns].fillna('NAN')

# data for catboost
X_train=X_all.iloc[:len(train_data),:]
X_test=X_all.iloc[len(train_data):]

# data for logreg
X_train_lor=X_train.copy()
for col in num_columns:
    X_train_lor[col]=X_train_lor[col].fillna(X_train_lor[col].median())
X_train_lor=pd.get_dummies(X_train_lor)

# 划分训练集、测试集
X_trn, X_val, y_trn, y_val = train_test_split(X_train, 
                                              y_train, 
                                              test_size=.2, 
                                              random_state=4321)
X_trn_lor = X_train_lor.loc[X_trn.index,:]
X_val_lor = X_train_lor.loc[X_val.index,:]


cat_features=[X_trn.columns.to_list().index(col) for col in cat_columns]

# catboost训练，迭代可能比较慢，可以设置iterations次数
%time cat = CatBoostClassifier(iterations=5, depth=5,cat_features=[0,1,6],silent=False).fit(X_trn,y_trn)

# 预测概率
probas_cat = pd.Series(cat.predict_proba(X_all)[:,1],index=X_all.index)
print('accuracy:',accuracy_score(y_val,cat.predict(X_val)))

# 获取全样本shap值
shap_df = cat.get_feature_importance(data = Pool(X_all, cat_features=cat_features), 
                                  type = 'ShapValues')
shap_df = pd.DataFrame(shap_df[:,:-1], columns = X_all.columns, index = X_all.index)
# shap_sum = shap_df.sum(axis = 1)
# shap_df

# shap -> prob映射
from util import shap2deltaprob_v2
shap_temp = shap2deltaprob_v2(shap_df,probas_cat)

shap_temp = shap_temp.head().applymap(lambda x:('+'if x>0 else '')+str(round(x*100,2))+'%')

shap_temp.style.apply(lambda x: ["background:orangered" if float(v[:-1])<0 else "background:lightgreen"
                                for v in x], axis = 1)


# 影响分析
from util import partial_deltaprob_v2 

# 单特征分析
shap_temp = shap2deltaprob_v2(shap_df, 
                   probas_xgb,
                   func_shap2probas = 'interp1d')

feature = 'Sex' # 一个特征
dp_col = shap_temp
out = partial_deltaprob_v2(feature, X_all, dp_col, cutoffs = None )

out = partial_deltaprob_v2('Pclass', X_all, dp_col, cutoffs = None )

out = partial_deltaprob_v2('Fare', X_all, dp_col,
                               cutoffs = [0,25,50,75,100,X_train['Fare'].max()])


plot_df(out)

# 特征交叉分析 - 分组汇总,不封装了...




