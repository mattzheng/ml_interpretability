# ml_interpretability

机器学习模型的可解释性。



安装相关依赖:
```
pip install xgboost
pip install catboost
```

下载titanic数据

```
!kaggle competitions download -c titanic
```

不过,老是报错,就直接随便从网上搜索下载了...


## shap值映射向概率
借助的是一元线性插值的方式，参考：
[Scipy Tutorial-插值interp1d](http://liao.cpython.org/scipytutorial10/)


## 相关报错：

```
ModuleNotFoundError: No module named 'sklearn.impute'
```
sklearn版本不一致,需要升级到0.20以上
```
pip install --upgrade scikit-learn
```



