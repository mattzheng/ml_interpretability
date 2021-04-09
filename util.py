# -*- coding: utf-8 -*-


from scipy.interpolate import interp1d
#probas_cat = pd.Series(cat.predict_proba(X_all)[:,1],index=X_all.index)

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
    



