[![PyPI version](https://img.shields.io/pypi/v/vecstack.svg?colorB=4cc61e)](https://pypi.python.org/pypi/vecstack)
[![PyPI license](https://img.shields.io/pypi/l/vecstack.svg)](https://github.com/vecxoz/vecstack/blob/master/LICENSE.txt)

# vecstack
Python package for stacking (machine learning technique)  
Convenient way to automate OOF computation, prediction and bagging using any number of models  
***Note:*** `OOF` is also known as `out-of-fold predictions`, `OOF features`, `stacked features`, `stacking features`, etc.
* Easy to use. Perform stacking in a [single line](https://github.com/vecxoz/vecstack#usage)
* Use any sklearn-like models
* Perform [classification and regression](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L160) tasks
* <sup>**NEW**</sup> Predict [probabilities](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L202) in classification task
* <sup>**NEW**</sup> [Modes](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L187): compute only what you need (only OOF, only predictions, both, etc.)
* <sup>**NEW**</sup> [Save](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L207) resulting arrays and log with model parameters
* Apply any [user-defined transformations](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L164) for target and prediction
* Python 2, Python 3
* Win, Linux, Mac
* [MIT license](https://github.com/vecxoz/vecstack/blob/master/LICENSE.txt)
* Depends on **numpy**, **scipy**, **scikit-learn>=18.0**

# Get started
* [Installation guide](https://github.com/vecxoz/vecstack#installation)
* [Usage](https://github.com/vecxoz/vecstack#usage)
* Examples:
    * [regression](https://github.com/vecxoz/vecstack/blob/master/examples/01_regression.ipynb)
    * [classification with class labels](https://github.com/vecxoz/vecstack/blob/master/examples/02_classification_with_class_labels.ipynb)
* Explanation of [**stacking concept**](https://github.com/vecxoz/vecstack#stacking-concept) with pictures
* You can also look at detailed [parameter description](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L136) or just type ```>>>help(stacking)```

# Installation

***Note:*** On Linux don't forget to use `pip/pip3` (or `python/python3`) to install package for desired version  

* ***Classic 1st time installation (recommended):*** 
    * `pip install vecstack`
* Install for current user only (if you have some troubles with write permission):
    * `pip install --user vecstack`
* If your PATH doesn't work: 
    * `/usr/bin/python -m pip install vecstack`
    * `C:/Python36/python -m pip install vecstack`
* If you got package archive:
    * `pip install vecstack.zip`
* Without PIP (if you're inside unpacked archive):
    * `python setup.py install`
* Upgrade vecstack and all dependencies:
    * `pip install --upgrade vecstack`
* Upgrade vecstack WITHOUT upgrading dependencies:
    * `pip install --upgrade --no-deps vecstack`
* Upgrade directly from GitHub WITHOUT upgrading dependencies:
    * `pip install --upgrade --no-deps https://github.com/vecxoz/vecstack/archive/master.zip`
* Uninstall
    * `pip uninstall vecstack`

# Usage
```python
from vecstack import stacking

# Get your data

# Initialize 1st level models

# Get your stacking features in a single line
S_train, S_test = stacking(models, X_train, y_train, X_test, regression = True, verbose = 2)

# Use 2nd level model with stacking features
```

# Complete examples

## Regression

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from vecstack import stacking

# Load demo data
boston = load_boston()
X, y = boston.data, boston.target

# Make train/test split
# As usual in machine learning task we have X_train, y_train, and X_test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size = 0.2, random_state = 0)

# Caution! All models and parameter values are just 
# demonstrational and shouldn't be considered as recommended.
# Initialize 1st level models.
models = [
    ExtraTreesRegressor(random_state = 0, n_jobs = -1, 
        n_estimators = 100, max_depth = 3),
        
    RandomForestRegressor(random_state = 0, n_jobs = -1, 
        n_estimators = 100, max_depth = 3),
        
    XGBRegressor(seed = 0, n_jobs = -1, learning_rate = 0.1, 
        n_estimators = 100, max_depth = 3)]
    
# Compute stacking features
S_train, S_test = stacking(models, X_train, y_train, X_test, 
    regression = True, metric = mean_absolute_error, n_folds = 4, 
    shuffle = True, random_state = 0, verbose = 2)

# Initialize 2nd level model
model = XGBRegressor(seed = 0, n_jobs = -1, learning_rate = 0.1, 
    n_estimators = 100, max_depth = 3)
    
# Fit 2nd level model
model = model.fit(S_train, y_train)

# Predict
y_pred = model.predict(S_test)

# Final prediction score
print('Final prediction score: [%.8f]' % mean_absolute_error(y_test, y_pred))
```

```
task:   [regression]
metric: [mean_absolute_error]

model 0: [ExtraTreesRegressor]
    fold 0: [3.20733439]
    fold 1: [2.87943130]
    fold 2: [2.53026486]
    fold 3: [2.83618694]
    ----
    MEAN:   [2.86330437]

model 1: [RandomForestRegressor]
    fold 0: [3.11110485]
    fold 1: [2.78404210]
    fold 2: [2.55707729]
    fold 3: [2.32209992]
    ----
    MEAN:   [2.69358104]

model 2: [XGBRegressor]
    fold 0: [2.40318939]
    fold 1: [2.37286982]
    fold 2: [1.89121530]
    fold 3: [1.95382831]
    ----
    MEAN:   [2.15527571]
    
Final prediction score: [2.78409065]
```

## Classification

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from vecstack import stacking

# Load demo data
iris = load_iris()
X, y = iris.data, iris.target

# Make train/test split
# As usual in machine learning task we have X_train, y_train, and X_test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size = 0.2, random_state = 0)

# Caution! All models and parameter values are just 
# demonstrational and shouldn't be considered as recommended.
# Initialize 1st level models.
models = [
    ExtraTreesClassifier(random_state = 0, n_jobs = -1, 
        n_estimators = 100, max_depth = 3),
        
    RandomForestClassifier(random_state = 0, n_jobs = -1, 
        n_estimators = 100, max_depth = 3),
        
    XGBClassifier(seed = 0, n_jobs = -1, learning_rate = 0.1, 
        n_estimators = 100, max_depth = 3)]
    
# Compute stacking features
S_train, S_test = stacking(models, X_train, y_train, X_test, 
    regression = False, metric = accuracy_score, n_folds = 4, 
    stratified = True, shuffle = True, random_state = 0, verbose = 2)

# Initialize 2nd level model
model = XGBClassifier(seed = 0, n_jobs = -1, learning_rate = 0.1, 
    n_estimators = 100, max_depth = 3)
    
# Fit 2nd level model
model = model.fit(S_train, y_train)

# Predict
y_pred = model.predict(S_test)

# Final prediction score
print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred))
```

```
task:   [classification]
metric: [accuracy_score]

model 0: [ExtraTreesClassifier]
    fold 0: [0.93548387]
    fold 1: [0.96666667]
    fold 2: [1.00000000]
    fold 3: [0.89655172]
    ----
    MEAN:   [0.95000000]

model 1: [RandomForestClassifier]
    fold 0: [0.87096774]
    fold 1: [0.96666667]
    fold 2: [1.00000000]
    fold 3: [0.93103448]
    ----
    MEAN:   [0.94166667]

model 2: [XGBClassifier]
    fold 0: [0.83870968]
    fold 1: [0.93333333]
    fold 2: [1.00000000]
    fold 3: [0.93103448]
    ----
    MEAN:   [0.92500000]
    
Final prediction score: [0.96666667]
```

# Stacking concept

1. We want to predict train and test sets with some 1st level model(s), and then use this predictions as features for 2nd level model.  
2. Any model can be used as 1st level model or 2nd level model.
3. To avoid overfitting (for train set) we use cross-validation technique and in each fold we predict out-of-fold part of train set.
4. The common practice is to use from 3 to 10 folds.
5. In each fold we predict full test set, so after completion of all folds we need to find mean (mode) of all test set predictions made in each fold. (Alternatively we can fit model on full train set and predict test set once. This approach takes more time because we need to perform one additional fitting, but may give higher test accuracy because we can use all train data for fitting.)
6. As an example we look at stacking implemented with single 1st level model and 3-fold cross-validation.
7. Three pictures below describe three folds of cross-validation. After completion of all three folds we get single train feature and single test feature to use with 2nd level model.
8. We can repeat this cycle using other 1st level models to get more features for 2nd level model.
9. At the bottom you can see [GIF animation](https://github.com/vecxoz/vecstack/blob/master/README.md#animation).

***
![stack1](https://github.com/vecxoz/vecstack/raw/master/pic/dia1.png "Fold 1 of 3")
***
![stack2](https://github.com/vecxoz/vecstack/raw/master/pic/dia2.png "Fold 2 of 3")
***
![stack3](https://github.com/vecxoz/vecstack/raw/master/pic/dia3.png "Fold 3 of 3")
***

# Animation
![animation](https://github.com/vecxoz/vecstack/raw/master/pic/dia.gif "Animation")
