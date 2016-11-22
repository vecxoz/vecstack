"""Python package for stacking (machine learning technique)

Find out how to use:
>>>from vecstack import stacking
>>>help(stacking)

MIT License

Copyright (c) 2016 vecxoz
Email: vecxoz@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

import numpy as np
import scipy.stats as st
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def transformer(y, func=None):
    """Transforms target variable and prediction"""
    if func is None:
        return y
    else:
        return func(y)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def stacking(models, X_train, y_train, X_test, regression=True,
             transform_target=None, transform_pred=None,
             metric=None, n_folds=4, stratified=False,
             shuffle=False, random_state=0, verbose=0):
    """Function 'stacking' takes train data, test data and list of 1-st level
    models, and returns stacking features, which can be used with 2-nd level model.
    
    Complete examples and stacking concept - see below.
    
    Parameters
    ----------
    models : list 
        List of 1-st level models. You can use any models that follow sklearn
        convention i.e. accept numpy arrays and have methods 'fit' and 'predict'.
        
    X_train : numpy array or sparse matrix of shape [n_train_samples, n_features]
        Training data
    
    y_train : numpy 1d array
        Target values
        
    X_test : numpy array or sparse matrix of shape [n_test_samples, n_features]
        Test data
        
    regression : boolean, default True
        If True - perform stacking for regression task, 
        if False - perform stacking for classification task
        
    transform_target : callable, default None
        Function to transform target variable.
        If None - transformation is not used.
        For example, for regression task (if target variable is skewed)
            you can use transformation like numpy.log.
            Set transform_target = numpy.log
        Usually you want to use respective backward transformation 
            for prediction like numpy.exp.
            Set transform_pred = numpy.exp
        Caution! Some transformations may give inapplicable results. 
            For example, if target variable contains zeros, numpy.log 
            gives you -inf. In such case you can use appropriate 
            transformation like numpy.log1p and respective
            backward transformation like numpy.expm1
        
    transform_pred : callable, default None
        Function to transform prediction.
        If None - transformation is not used.
        If you use transformation for target variable (transform_target)
            like numpy.log, then using transform_pred you can specify 
            respective backward transformation like numpy.exp.
        Look at description of parameter transform_target
        
    metric : callable, default None
        Evaluation metric (score function) which is used to calculate 
        results of cross-validation.
        If None, then by default:
            sklearn.metrics.mean_absolute_error - for regression
            sklearn.metrics.accuracy_score - for classification
        You can use any appropriate sklearn metric or 
            define your own metric like shown below:
        
        def your_metric(y_true, y_pred):
            # calculate
            return result
        
    n_folds : int, default 4
        Number of folds in cross-validation
        
    stratified : boolean, default False, meaningful only for classification task
        If True - use stratified folds in cross-validation
        
    shuffle : boolean, default False
        Whether to perform a shuffle before cross-validation split
        
    random_state : int, default 0
        Random seed for shuffle
        
    verbose : int, default 0
        Level of verbosity.
        0 - show no messages
        1 - for each model show single mean score
        2 - for each model show score for each fold and mean score
        
        Caution. To calculate MEAN score across all folds 
        full train set prediction and full true target are used.
        So for some metrics (e.g. rmse) this value may not be equal 
        to mean of score values calculated for each fold.
        
    Returns
    -------
    S_train : numpy array of shape [n_train_samples, n_models]
        Stacking features for train set
        
    S_test : numpy array of shape [n_test_samples, n_models]
        Stacking features for test set
    
    Brief example (complete examples - see below)
    ---------------------------------------------
    from vecstack import stacking

    # Get your data

    # Initialize 1-st level models

    # Get your stacking features in a single line
    S_train, S_test = stacking(models, X_train, y_train, X_test, 
        regression = True, verbose = 2)

    # Use 2-nd level model with stacking features
    
    Stacking concept
    ----------------
    1. We want to predict train and test sets with some 1-st level model(s), 
       and then use this predictions as features for 2-nd level model.
    2. Any model can be used as 1-st level model or 2-nd level model.
    3. To avoid overfitting (for train set) we use cross-validation technique 
       and in each fold we predict out-of-fold part of train set.
    4. The common practice is to use from 3 to 10 folds.
    5. In each fold we predict full test set, so after completion of all folds 
       we need to find mean (mode) of all test set predictions made in each fold.
    
    You can find further stacking explanation with pictures at
    https://github.com/vecxoz/vecstack
    
    Complete examples
    -----------------
    
    Regression
    ----------
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
    
    
    Classification
    --------------
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
    """
    # Print type of task
    if regression and verbose > 0:
        print('task:   [regression]')
    elif not regression and verbose > 0:
        print('task:   [classification]')

    # Specify default metric for cross-validation
    if metric is None and regression:
        metric = mean_absolute_error
    elif metric is None and not regression:
        metric = accuracy_score
        
    # Print metric
    if verbose > 0:
        print('metric: [%s]\n' % metric.__name__)
        
    # Split indices to get folds (stratified can be used only for classification)
    if stratified and not regression:
        kf = StratifiedKFold(y_train, n_folds, shuffle = shuffle, random_state = random_state)
    else:
        kf = KFold(len(y_train), n_folds, shuffle = shuffle, random_state = random_state)

    # Create empty numpy arrays for stacking features
    S_train = np.zeros((X_train.shape[0], len(models)))
    S_test = np.zeros((X_test.shape[0], len(models)))
    
    # Loop across models
    for model_counter, model in enumerate(models):
        if verbose > 0:
            print('model %d: [%s]' % (model_counter, model.__class__.__name__))
            
        # Create empty numpy array, which will contain temporary predictions for test set made in each fold
        S_test_temp = np.zeros((X_test.shape[0], len(kf)))
        
        # Loop across folds
        for fold_counter, (tr_index, te_index) in enumerate(kf):
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            
            # Fit 1-st level model
            model = model.fit(X_tr, transformer(y_tr, func = transform_target))
            # Predict out-of-fold part of train set
            S_train[te_index, model_counter] = transformer(model.predict(X_te), func = transform_pred)
            # Predict full test set
            S_test_temp[:, fold_counter] = transformer(model.predict(X_test), func = transform_pred)
            
            if verbose > 1:
                print('    fold %d: [%.8f]' % (fold_counter, metric(y_te, S_train[te_index, model_counter])))
                
        # Compute mean or mode of predictions for test set
        if regression:
            S_test[:, model_counter] = np.mean(S_test_temp, axis = 1)
        else:
            S_test[:, model_counter] = st.mode(S_test_temp, axis = 1)[0].ravel()
            
        if verbose > 0:
            print('    ----')
            print('    MEAN:   [%.8f]\n' % (metric(y_train, S_train[:, model_counter])))

    return (S_train, S_test)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

