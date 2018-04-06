[![PyPI version](https://img.shields.io/pypi/v/vecstack.svg?colorB=4cc61e)](https://pypi.python.org/pypi/vecstack)
[![PyPI license](https://img.shields.io/pypi/l/vecstack.svg)](https://github.com/vecxoz/vecstack/blob/master/LICENSE.txt)
[![Build Status](https://travis-ci.org/vecxoz/vecstack.svg?branch=master)](https://travis-ci.org/vecxoz/vecstack)
[![Coverage Status](https://coveralls.io/repos/github/vecxoz/vecstack/badge.svg?branch=master)](https://coveralls.io/github/vecxoz/vecstack?branch=master)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/vecstack.svg)](https://pypi.python.org/pypi/vecstack/)

# vecstack
Python package for stacking featuring lightweight ***functional API*** and fully compatible ***scikit-learn API***  
Convenient way to automate OOF computation, prediction and bagging using any number of models  

* [Functional API](https://github.com/vecxoz/vecstack#usage-functional-api):
    * Minimalistic. Get your stacked features in a single line
    * RAM-friendly. The lowest possible memory consumption
    * Kaggle-ready. Stacked features and hyperparameters from each run can be [automatically saved](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L209) in files. No more mess at the end of the competition.  [Log example](https://github.com/vecxoz/vecstack/blob/master/examples/03_log_example.txt)
* [Scikit-learn API](https://github.com/vecxoz/vecstack#usage-scikit-learn-api):
    * Standardized. Fully scikit-learn compatible transformer class exposing `fit` and `transform` methods
    * Pipeline-certified. Implement and deploy [multilevel stacking](https://github.com/vecxoz/vecstack/blob/master/examples/04_sklearn_api_regression_pipeline.ipynb) like it's no big deal using `sklearn.pipeline.Pipeline` 
    * And of course `FeatureUnion` and `GridSearchCV` are also invited to the party
* Overall specs:
    * Use any sklearn-like estimators
    * Perform [classification and regression](https://github.com/vecxoz/vecstack/blob/master/vecstack/coresk.py#L83) tasks
    * Predict [class labels or probabilities](https://github.com/vecxoz/vecstack/blob/master/vecstack/coresk.py#L119) in classification task
    * Apply any [user-defined metric](https://github.com/vecxoz/vecstack/blob/master/vecstack/coresk.py#L124)
    * Apply any [user-defined transformations](https://github.com/vecxoz/vecstack/blob/master/vecstack/coresk.py#L87) for target and prediction
    * Python 2, Python 3
    * Win, Linux, Mac
    * [MIT license](https://github.com/vecxoz/vecstack/blob/master/LICENSE.txt)
    * Depends on **numpy**, **scipy**, **scikit-learn>=18.0**

# Get started
* [Installation guide](https://github.com/vecxoz/vecstack#installation)
* Usage:
    * [Functional API](https://github.com/vecxoz/vecstack#usage-functional-api)
    * [Scikit-learn API](https://github.com/vecxoz/vecstack#usage-scikit-learn-api)
* Tutorials:
    * [Stacking concept + Pictures + Stacking implementation from scratch](https://github.com/vecxoz/vecstack/blob/master/examples/00_stacking_concept_pictures_code.ipynb)
* Examples:
    * Functional API:
        * [Regression](https://github.com/vecxoz/vecstack/blob/master/examples/01_regression.ipynb)
        * [Classification with class labels](https://github.com/vecxoz/vecstack/blob/master/examples/02_classification_with_class_labels.ipynb)
        * [Classification with probabilities + Detailed workflow](https://github.com/vecxoz/vecstack/blob/master/examples/03_classification_with_proba_detailed_workflow.ipynb)
    * Scikit-learn API:
        * [Regression + Multilevel stacking using Pipeline](https://github.com/vecxoz/vecstack/blob/master/examples/04_sklearn_api_regression_pipeline.ipynb)
* Documentation:
    * [Functional API](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L133) or type ```>>> help(stacking)```
    * [Scikit-learn API](https://github.com/vecxoz/vecstack/blob/master/vecstack/coresk.py#L64) or type ```>>> help(StackingTransformer)```

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

# Usage. Functional API
```python
from vecstack import stacking

# Get your data

# Initialize 1st level estimators
models = [LinearRegression(),
          Ridge(random_state=0)]

# Get your stacked features in a single line
S_train, S_test = stacking(models, X_train, y_train, X_test, regression=True, verbose=2)

# Use 2nd level estimator with stacked features
```

# Usage. Scikit-learn API
```python
from vecstack import StackingTransformer

# Get your data

# Initialize 1st level estimators
estimators = [('lr', LinearRegression()),
              ('ridge', Ridge(random_state=0))]
              
# Initialize StackingTransformer
stack = StackingTransformer(estimators, regression=True, verbose=2)

# Fit
stack = stack.fit(X_train, y_train)

# Get your stacked features
S_train = stack.transform(X_train)
S_test = stack.transform(X_test)

# Use 2nd level estimator with stacked features
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
