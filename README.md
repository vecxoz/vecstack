# vecstack
Python package for stacking (machine learning technique)
* Easy to use. Perform stacking in a [single line](https://github.com/vecxoz/vecstack#brief-example)
* Use any sklearn-like models
* Perform classification and regression tasks
* Apply any user-defined transformations for target and prediction
* Python 2, Python 3
* Win, Linux, Mac
* [MIT license](https://github.com/vecxoz/vecstack/blob/master/LICENSE.txt)
* Depends on **numpy**, **scipy**, **scikit-learn**

Below you can find
* [Installation](https://github.com/vecxoz/vecstack#installation) guide
* [Brief example](https://github.com/vecxoz/vecstack#brief-example)
* Complete examples for [regression](https://github.com/vecxoz/vecstack#regression) and [classification](https://github.com/vecxoz/vecstack#classification)
* Explanation of [**stacking concept**](https://github.com/vecxoz/vecstack#stacking-concept) with pictures

You can also look at detailed [parameter description](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L61) or just type ```>>>help(stacking)```

# Installation

### First way
* From the command line run ```pip install vecstack```

### Second way
* Download package file from [PyPI](https://pypi.python.org/pypi/vecstack)  
* From the command line go to the directory where you've downloaded package file
* Run ```python -m pip install vecstack-0.1.zip```

# Brief example
```python
from vecstack import stacking

# Get your data

# Initialize 1-st level models

# Get your stacking features in a single line
S_train, S_test = stacking(models, X_train, y_train, X_test, regression = True, verbose = 2)

# Use 2-nd level model with stacking features
```

# Complete examples

## Regression

```python
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
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
# Initialize 1-st level models.
models = [
    ExtraTreesRegressor(random_state = 0, n_jobs = -1, 
        n_estimators = 100, max_depth = 3),
        
    RandomForestRegressor(random_state = 0, n_jobs = -1, 
        n_estimators = 100, max_depth = 3),
        
    XGBRegressor(seed = 0, nthread = -1, learning_rate = 0.1, 
        n_estimators = 100, max_depth = 3)]
    
# Compute stacking features
S_train, S_test = stacking(models, X_train, y_train, X_test, 
    regression = True, metric = mean_absolute_error, n_folds = 4, 
    shuffle = True, random_state = 0, verbose = 2)

# Initialize 2-nd level model
model = XGBRegressor(seed = 0, nthread = -1, learning_rate = 0.1, 
    n_estimators = 100, max_depth = 3)
    
# Fit 2-nd level model
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
from sklearn.cross_validation import train_test_split
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
# Initialize 1-st level models.
models = [
    ExtraTreesClassifier(random_state = 0, n_jobs = -1, 
        n_estimators = 100, max_depth = 3),
        
    RandomForestClassifier(random_state = 0, n_jobs = -1, 
        n_estimators = 100, max_depth = 3),
        
    XGBClassifier(seed = 0, nthread = -1, learning_rate = 0.1, 
        n_estimators = 100, max_depth = 3)]
    
# Compute stacking features
S_train, S_test = stacking(models, X_train, y_train, X_test, 
    regression = False, metric = accuracy_score, n_folds = 4, 
    stratified = True, shuffle = True, random_state = 0, verbose = 2)

# Initialize 2-nd level model
model = XGBClassifier(seed = 0, nthread = -1, learning_rate = 0.1, 
    n_estimators = 100, max_depth = 3)
    
# Fit 2-nd level model
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

1. We want to predict train and test sets with some 1-st level model(s), and then use this predictions as features for 2-nd level model.  
2. Any model can be used as 1-st level model or 2-nd level model.
3. To avoid overfitting (for train set) we use cross-validation technique and in each fold we predict out-of-fold part of train set.
4. The common practice is to use from 3 to 10 folds.
5. In each fold we predict full test set, so after completion of all folds we need to find mean (mode) of all test set predictions made in each fold.
6. As an example we look at stacking implemented with single 1-st level model and 3-fold cross-validation.
7. Tree pictures below describe three folds of cross-validation. After completion of all three folds we get single train feature and single test feature to use with 2-nd level model.
8. We can repeat this cycle using other 1-st level models to get more features for 2-nd level model.
9. At the bottom you can see [GIF animation](https://github.com/vecxoz/vecstack/blob/master/README.md#animation).

***
![stack1](https://github.com/vecxoz/vecstack/blob/master/pic/dia1.png "Fold 1 of 3")
***
![stack2](https://github.com/vecxoz/vecstack/blob/master/pic/dia2.png "Fold 2 of 3")
***
![stack3](https://github.com/vecxoz/vecstack/blob/master/pic/dia3.png "Fold 3 of 3")
***

# Animation
![animation](https://github.com/vecxoz/vecstack/blob/master/pic/dia.gif "Animation")
