[![PyPI version](https://img.shields.io/pypi/v/vecstack.svg?colorB=4cc61e)](https://pypi.python.org/pypi/vecstack)
[![PyPI license](https://img.shields.io/pypi/l/vecstack.svg)](https://github.com/vecxoz/vecstack/blob/master/LICENSE.txt)
[![Build Status](https://travis-ci.org/vecxoz/vecstack.svg?branch=master)](https://travis-ci.org/vecxoz/vecstack)
[![Coverage Status](https://coveralls.io/repos/github/vecxoz/vecstack/badge.svg?branch=master)](https://coveralls.io/github/vecxoz/vecstack?branch=master)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/vecstack.svg)](https://pypi.python.org/pypi/vecstack/)

# vecstack
Python package for stacking (machine learning technique)  
Convenient way to automate OOF computation, prediction and bagging using any number of models  
***Note:*** `OOF` is also known as `out-of-fold predictions`, `OOF features`, `stacked features`, `stacking features`, etc.
* Easy to use. Perform stacking in a [single line](https://github.com/vecxoz/vecstack#usage)
* Use any sklearn-like models
* Perform [classification and regression](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L160) tasks
* <sup>**NEW**</sup> Predict [probabilities](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L202) in classification task
* <sup>**NEW**</sup> [Modes](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L187): compute only what you need (only OOF, only predictions, both, etc.)
* <sup>**NEW**</sup> [Save](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L207) resulting arrays and log with model parameters ([log example](https://github.com/vecxoz/vecstack/blob/master/examples/03_log_example.txt))
* Apply any [user-defined transformations](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L164) for target and prediction
* Python 2, Python 3
* Win, Linux, Mac
* [MIT license](https://github.com/vecxoz/vecstack/blob/master/LICENSE.txt)
* Depends on **numpy**, **scipy**, **scikit-learn>=18.0**

# Get started
* [Installation guide](https://github.com/vecxoz/vecstack#installation)
* [Usage](https://github.com/vecxoz/vecstack#usage)
* Tutorials:
    * [Stacking concept + Pictures + Stacking implementation from scratch](https://github.com/vecxoz/vecstack/blob/master/examples/00_stacking_concept_pictures_code.ipynb)
* Examples:
    * [Regression](https://github.com/vecxoz/vecstack/blob/master/examples/01_regression.ipynb)
    * [Classification with class labels](https://github.com/vecxoz/vecstack/blob/master/examples/02_classification_with_class_labels.ipynb)
    * [Classification with probabilities + Detailed workflow](https://github.com/vecxoz/vecstack/blob/master/examples/03_classification_with_proba_detailed_workflow.ipynb)
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

1. We want to predict train set and test set with some 1st level model(s), and then use these predictions as features for 2nd level model(s).  
2. Any model can be used as 1st level model or 2nd level model.
3. To avoid overfitting (for train set) we use cross-validation technique and in each fold we predict out-of-fold (OOF) part of train set.
4. The common practice is to use from 3 to 10 folds.
5. Predict test set:
   * **Variant A:** In each fold we predict test set, so after completion of all folds we need to find mean (mode) of all temporary test set predictions made in each fold. 
   * **Variant B:** We do not predict test set during cross-validation cycle. After completion of all folds we perform additional step: fit model on full train set and predict test set once. This approach takes more time because we need to perform one additional fitting.
6. As an example we look at stacking implemented with single 1st level model and 3-fold cross-validation.
7. Pictures:
   * **Variant A:** Three pictures describe three folds of cross-validation. After completion of all three folds we get single train feature and single test feature to use with 2nd level model. 
   * **Variant B:** First three pictures describe three folds of cross-validation (like in Variant A) to get single train feature and fourth picture describes additional step to get single test feature.
8. We can repeat this cycle using other 1st level models to get more features for 2nd level model.
9. You can also look at animation of [Variant A](https://github.com/vecxoz/vecstack#variant-a-animation) and [Variant B](https://github.com/vecxoz/vecstack#variant-b-animation).

# Variant A

![Fold 1 of 3](https://github.com/vecxoz/vecstack/raw/master/pic/dia1.png "Fold 1 of 3")
***
![Fold 2 of 3](https://github.com/vecxoz/vecstack/raw/master/pic/dia2.png "Fold 2 of 3")
***
![Fold 3 of 3](https://github.com/vecxoz/vecstack/raw/master/pic/dia3.png "Fold 3 of 3")

# Variant A. Animation

![Variant A. Animation](https://github.com/vecxoz/vecstack/raw/master/pic/animation1.gif "Variant A. Animation")

# Variant B

![Step 1 of 4](https://github.com/vecxoz/vecstack/raw/master/pic/dia4.png "Step 1 of 4")
***
![Step 2 of 4](https://github.com/vecxoz/vecstack/raw/master/pic/dia5.png "Step 2 of 4")
***
![Step 3 of 4](https://github.com/vecxoz/vecstack/raw/master/pic/dia6.png "Step 3 of 4")
***
![Step 4 of 4](https://github.com/vecxoz/vecstack/raw/master/pic/dia7.png "Step 4 of 4")

# Variant B. Animation

![Variant B. Animation](https://github.com/vecxoz/vecstack/raw/master/pic/animation2.gif "Variant B. Animation")
