'''

参考文章:[SHAP的理解与应用](https://zhuanlan.zhihu.com/p/103370775)
里面有专门处理类别变量的方式，不过文章中的结论是，是否one-hot处理，差别蛮大，貌似我自己测试，没有差别，
可能是我哪一步出错了...没细究...

'''


import lightgbm as lgb
import shap
import pandas as pd


X,y = shap.datasets.adult()
X_display,y_display = shap.datasets.adult(display=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

model = lgb.LGBMClassifier(random_state=1234)
model.fit(X_train, y_train)


explainer = shap.TreeExplainer(model)
# 最新版本的shap对于LGBMClassifier得到的shap_values为两个数组的列表，即两个分类的结果，这里使用分类1的结果
shap_values = explainer.shap_values(X)

if len(shap_values) == 2:
    shap_values = shap_values[1]
# 如果报错，更新lightgbm
# shap_interaction_values = explainer.shap_interaction_values(X)

# 如果explainer.expected_value得到两个值，则应该为explainer.expected_value[1]
print('解释模型的常数:', explainer.expected_value)
print('训练样本预测值的log odds ratio的均值:', np.log(model.predict_proba(X_train)[:, 1]/ (1 - model.predict_proba(X_train)[:, 1])).mean())
print('常数与归因值之和:', explainer.expected_value + shap_values[0].sum())
print('预测值:', model.predict_proba(X.iloc[0:1])[:, 1][0])
print('预测值的log odds ratio:', np.log(model.predict_proba(X.iloc[0:1])[:, 1][0] / (1 - model.predict_proba(X.iloc[0:1])[:, 1][0])))

# shap值原始的特征
# feature_importance = pd.DataFrame()
# feature_importance['feature'] = X.columns
# feature_importance['importance'] = np.abs(shap_values).mean(0)
# feature_importance.sort_values('importance', ascending=False)

feature_importance = pd.DataFrame()
feature_importance['feature'] = X.columns
feature_importance['importance'] = np.abs(shap_values).mean(0)
feature_importance.sort_values('importance', ascending=False)

'''
           feature  importance
5     Relationship    1.054445
0              Age    0.855286
8     Capital Gain    0.600313
2    Education-Num    0.456429
10  Hours per week    0.330449
4       Occupation    0.279986
3   Marital Status    0.254001
7              Sex    0.157957
9     Capital Loss    0.147177
1        Workclass    0.057955
6             Race    0.047785
11         Country    0.020872


'''

def onehot_pipeline(model, X_train, y_train, char_cols=None, num_fillna=None, char_fillna=None):
    '''
    传入带有参数的模型,封装成类别特征one-hot的pipline
    ————————————————————————————————————
    入参:
        model:带有参数的模型
        X_train:训练集的特征,pd.DataFrame格式
        Y_train:训练集的目标
        char_cols:类别特征的列表,如不传入自动根据数据类型获取
        num_fillna:数值特征的缺失填充值,可支持不填充
        char_fillna:类别特征的缺失填充值,可支持不传入,但模型会自动填充null用于one-hot
    出参:
        pipeline:封装好mapper和model的pipeline,并训练完成
    '''
    from sklearn_pandas import DataFrameMapper
    from sklearn.preprocessing import OneHotEncoder
    from sklearn2pmml.decoration import ContinuousDomain, CategoricalDomain
    from sklearn.pipeline import Pipeline
    
    if not char_cols:
        col_types = X_train.dtypes
        char_cols = list(col_types[col_types.apply(lambda x: 'int' not in str(x) and 'float' not in str(x))].index)
    num_cols = list(set(X_train.columns) - set(char_cols))
    
    if not isinstance(char_fillna, str):
        char_fillna = 'null'
        
    mapper = DataFrameMapper(
        [(num_cols, ContinuousDomain(missing_value_replacement=num_fillna, with_data=False))] +
        [([char_col], [CategoricalDomain(missing_value_replacement=char_fillna, invalid_value_treatment='as_is'),
                       OneHotEncoder(handle_unknown='ignore')])
         for char_col in char_cols]
    )
    
    pipeline = Pipeline(steps=[('mapper', mapper), ('model', model)])
    pipeline.fit(X_train, y_train)
    
    return pipeline


def pipeline_shap(pipeline, X_train, y_train, interaction=False, sample=None):
    '''
    获取由onehot_pipeline返回的pipeline的shap值
    ————————————————————————————————————
    入参:
        pipeline:onehot_pipeline的返回对象
        X_train:训练集的特征,pd.DataFrame格式
        Y_train:训练集的目标
        interaction:是否返回shap interaction values
        sample:抽样数int或抽样比例float,不传入则不抽样
    出参:
        feature_values:如传入sample则是抽样后的X_train,否则为X_train
        shap_values:pd.DataFrame格式shap values,如interaction传入True,则为shap interaction values
    '''
    import shap
    
    if isinstance(sample, int):
        feature_values = X_train.sample(n=sample)
    elif isinstance(sample, float):
        feature_values = X_train.sample(frac=sample)
    else:
        feature_values = X_train
        
    mapper = pipeline.steps[0][1]
    model = pipeline._final_estimator
    sort_cols, onehot_cols = [], []
    for i in mapper.features:
        sort_cols += i[0]
        if 'OneHot' in str(i[1]):
            onehot_cols += i[0]
    feature_values = feature_values[sort_cols]
    
    mapper.fit(X_train)
    X_train_mapper = mapper.transform(X_train)
    feature_values_mapper = mapper.transform(feature_values)
    model.fit(X_train_mapper, y_train)
    
    shap_values = pd.DataFrame(index=feature_values.index, columns=feature_values.columns)
    explainer = shap.TreeExplainer(model)
    if interaction:
        mapper_shap_values = explainer.shap_interaction_values(feature_values_mapper)
        col_index = 0
        for col in sort_cols:
            if col in onehot_cols:
                col_index_span = len(X_train[col].unique())
                shap_values[col] = mapper_shap_values[
                    :, col_index: col_index + col_index_span, col_index: col_index + col_index_span
                ].sum(2).sum(1)
                col_index += col_index_span
            else:
                shap_values[col] = mapper_shap_values[:, col_index, col_index]
                col_index += 1
    else:
        mapper_shap_values = explainer.shap_values(feature_values_mapper)
        if len(mapper_shap_values) == 2:
            mapper_shap_values = mapper_shap_values[1]
        col_index = 0
        for col in sort_cols:
            if col in onehot_cols:
                col_index_span = len(X_train[col].unique())
                shap_values[col] = mapper_shap_values[
                    :, col_index: col_index + col_index_span
                ].sum(1)
                col_index += col_index_span
            else:
                shap_values[col] = mapper_shap_values[:, col_index]
                col_index += 1
                
    return feature_values, shap_values



'''
经过onehot处理之后：

ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().

           feature  importance
3     Relationship    1.000000
6              Age    0.812198
9     Capital Gain    0.566544
0    Education-Num    0.432721
8   Hours per week    0.314213
5       Occupation    0.264969
10  Marital Status    0.240180
2              Sex    0.144359
7     Capital Loss    0.140176
4        Workclass    0.054945
11            Race    0.045173
1          Country    0.019816


'''
# 如果报错could not convert string to float，需更新skearn到0.20以上
pipeline = onehot_pipeline(model, X_train, y_train.astype(int))
feature_values, shap_values = pipeline_shap(pipeline, X_train, y_train)  
# 原作者多了一个X:shap_values = pipeline_shap(pipeline, X_train, y_train, X)
feature_importance = pd.DataFrame()
feature_importance['feature'] = shap_values.columns
feature_importance['importance'] = feature_importance['feature'].map(np.abs(shap_values).mean(0)) / np.abs(shap_values).mean(0).max()
feature_importance.sort_values('importance', ascending=False)


