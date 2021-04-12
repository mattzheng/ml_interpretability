# -*- coding: utf-8 -*-

import pandas as pd
from scipy.interpolate import interp1d
#probas_cat = pd.Series(cat.predict_proba(X_all)[:,1],index=X_all.index)
import numpy as np
def shap2deltaprob_v2(shap_df, 
                   probas,
                   func_shap2probas = 'interp1d'):
    '''
    shap值 -> 映射到概率上面
    
    
    map shap to Δ probabilities
    --- input ---
    :features: list of strings, names of features
    :shap_df: pd.DataFrame, dataframe containing shap values
    :shap_sum: pd.Series, series containing shap sum for each observation
    :probas: pd.Series, series containing predicted probability for each observation
    :func_shap2probas: function, maps shap to probability (for example interpolation function)
    --- output ---
    :out: pd.Series or pd.DataFrame, Δ probability for each shap value
    '''
    
    feat_columns = shap_df.columns.to_list() # 模型的特征名称
    shap_sum = shap_df.sum(axis = 1)
    
    if func_shap2probas == 'interp1d' :
        shap_sum_sort = shap_sum.sort_values()
        probas_cat_sort = probas[shap_sum_sort.index]
        intp = interp1d(shap_sum_sort,
                        probas_cat_sort, 
                        bounds_error = False, 
                        fill_value = (0, 1))
    
    # 1 feature
    if type(feat_columns) == str or len(feat_columns) == 1:
        return probas - (shap_sum - shap_df[feat_columns]).apply(intp)
    # more than 1 feature
    else:
        return shap_df[feat_columns].apply(lambda x: shap_sum - x).apply(intp)\
                .apply(lambda x: probas - x)
    

# dp = partial_deltaprob('Sex', X_all, shap_df, shap_sum, probas_cat, func_shap2probas=intp)
def partial_deltaprob_v2(feature, X, dp_col, cutoffs = None ):
    '''
    return univariate analysis (count, mean and standard deviation) of shap values based on the original feature
    --- input ---
    :feature: str, name of feature,单个样本
    :X: pd.Dataframe, shape (N, P),全量的X，不是训练集
    :dp_col：shap计算之后全样本值
    :cutoffs: list of floats, cutoffs for numerical features,是否去掉一些特征
    --- output ---
    :out: pd.DataFrame, shape (n_levels, 3)
    '''
    # dp_col = shap2deltaprob(feature, shap_df, shap_sum, probas, func_shap2probas)
    dp_col_mean = dp_col.mean()
    dp_col.columns = 'DP_' + dp_col.columns
    out = pd.concat([X[feature], dp_col], axis = 1)
    if cutoffs:
        intervals = pd.IntervalIndex.from_tuples(list(zip(cutoffs[:-1], cutoffs[1:])))
        out[feature] = pd.cut(out[feature], bins = intervals)
        out = out.dropna()   
    out = out.groupby(feature).describe().iloc[:, :3]
    out.columns = ['count', 'mean', 'std']
    out['std'] = out['std'].fillna(0)
    return out

import matplotlib.pyplot as plt

def plot_df(dp):
    # partial_deltaprob_v2 输出值进行画图
    plt.plot([0,len(dp)-1],[0,0],color='dimgray',zorder=3)
    plt.plot(range(len(dp)), dp['mean'], color = 'red', linewidth = 3, label = 'Avg effect',zorder=4)
    plt.fill_between(range(len(dp)), dp['mean'] + dp['std'], dp['mean'] - dp['std'],
                     color = 'lightskyblue', label = 'Avg effect +- StDev',zorder=1)
    yticks = list(np.arange(-.2,.41,.1))
    plt.yticks(yticks, [('+' if y > 0 else '') + '{0:.1%}'.format(y) for y in yticks], fontsize=13)
    plt.xticks(range(len(dp)), dp.index, fontsize=13)
    plt.ylabel('Effect on Predicted\nProbability of Survival',fontsize=13)
    plt.xlabel('Sex', fontsize=13)
    plt.title('Marginal effect of\nSex', fontsize=15)
    plt.gca().set_facecolor('lightyellow')
    plt.grid(zorder=2)