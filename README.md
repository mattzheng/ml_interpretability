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

## 关联文章

- [机器学习模型可解释性进行到底 —— 从SHAP值到预测概率（二）](https://mattzheng.blog.csdn.net/article/details/115556182)
- [机器学习模型可解释性进行到底 —— SHAP值理论（一）](https://blog.csdn.net/sinat_26917383/article/details/115400327)
- [机器学习模型可解释性进行到底 ——PDP&ICE图（三）](https://mattzheng.blog.csdn.net/article/details/115669705)

## 相关文件

- catboost_test.py - 测试catboost&shap值的demo
- xgboost_test.py - 测试xgb&shap值的demo
- pdd_ice_test.py - PDD / ICE的画图demo, **需要sklearn 0.24+**
- lightgbm_test.py - ligthgbm测试类别变量下的shap值demo

## 类别特征处理

- lightgbm_test.py - ligthgbm测试类别变量下的shap值demo

参考文章:[SHAP的理解与应用](https://zhuanlan.zhihu.com/p/103370775)
里面有专门处理类别变量的方式，不过文章中的结论是，是否one-hot处理，差别蛮大，貌似我自己测试，没有差别，
可能是我哪一步出错了...没细究...

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



