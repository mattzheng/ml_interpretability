
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split



# California Housing data preprocessing
cal_housing = fetch_california_housing()
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
y = cal_housing.target

y -= y.mean()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0
)

# Multi-layer perceptron
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.neural_network import MLPRegressor

print("Training MLPRegressor...")
tic = time()
est = make_pipeline(QuantileTransformer(),
                    MLPRegressor(hidden_layer_sizes=(50, 50),
                                 learning_rate_init=0.01,
                                 early_stopping=True))
est.fit(X_train, y_train)
print(f"done in {time() - tic:.3f}s")
print(f"Test R2 score: {est.score(X_test, y_test):.2f}")


from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence

# 部分依赖图（Partial Dependence Plot)
print('Computing partial dependence plots...')
tic = time()
features = ['MedInc', 'AveOccup', 'HouseAge', 'AveRooms']
display = plot_partial_dependence(
       est, X_train, features, kind="average", subsample=50,
       n_jobs=3, grid_resolution=20, random_state=0
)


print(f"done in {time() - tic:.3f}s")
display.figure_.suptitle(
    'Partial Dependence Plot\n'
)
display.figure_.subplots_adjust(hspace=0.3)

## 2D interaction plots
features = ['AveOccup', 'HouseAge', ('AveOccup', 'HouseAge')]
print('Computing partial dependence plots...')
tic = time()
_, ax = plt.subplots(ncols=3, figsize=(9, 4))
display = plot_partial_dependence(
    est, X_train, features, kind='average', n_jobs=3, grid_resolution=20,
    ax=ax,
)
print(f"done in {time() - tic:.3f}s")
display.figure_.suptitle(
    'Partial dependence of house value on non-location features\n'
    'for the California housing dataset, with Gradient Boosting'
)
display.figure_.subplots_adjust(wspace=0.4, hspace=0.3)


# 个体条件期望图（Individual Conditional Expectation Plot)
print('Computing partial dependence plots...')
tic = time()
features = ['MedInc', 'AveOccup', 'HouseAge', 'AveRooms']
display = plot_partial_dependence(
       est, X_train, features, kind="individual", subsample=50,
       n_jobs=3, grid_resolution=20, random_state=0
)
# average / individual 

print(f"done in {time() - tic:.3f}s")
display.figure_.suptitle(
    'Individual Conditional Expectation Plot\n'
)
display.figure_.subplots_adjust(hspace=0.3)

# both = PDP + ICE
print('Computing partial dependence plots...')
tic = time()
features = ['MedInc', 'AveOccup', 'HouseAge', 'AveRooms']
display = plot_partial_dependence(
       est, X_train, features, kind="both", subsample=50,
       n_jobs=3, grid_resolution=20, random_state=0
)
# average / individual 

print(f"done in {time() - tic:.3f}s")
display.figure_.suptitle(
    'Individual Conditional Expectation Plot\n'
)
display.figure_.subplots_adjust(hspace=0.3)

