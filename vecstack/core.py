"""Python package for stacking.
Author: <vecxoz@gmail.com>
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

def log1p(y, do = False):
    """
    Wrapper for numpy.log1p
    """
    if do:
        return np.log1p(y)
    else:
        return y
        
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
        
def expm1(y, do = False):
    """
    Wrapper for numpy.expm1
    """
    if do:
        return np.expm1(y)
    else:
        return y

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def stacking(models, X_train, y_train, X_test, regression = True, 
    log_transform = False, feval = None, n_folds = 4, stratified = False, 
    shuffle = False, random_state = 0, verbose = 0):
    """Function 'stacking' takes train data, test data and list of 1-st level
    models, and return stacking features, which can be used with 2-nd level model.
    
    Complete examples and stacking concept - see below.
    
    Parameters
    ----------
    models : list 
        List of 1-st level models. You can use any models that follow sklearn
        convention i.e. have methods 'fit' and 'predict'.
        
    X_train : numpy array or sparse matrix of shape [n_train_samples, n_features]
        Training data
    
    y_train : numpy 1d array
        Target values
        
    X_test : numpy array or sparse matrix of shape [n_test_samples, n_features]
        Test data
        
    regression : boolean, default True
        If True - perform stacking for regression task, 
        if False - perfrom stacking for classification task
        
    log_transform : boolean, default False, meaningful only for regression task
        If True - use numpy.log1p transform for target and then numpy.expm1 
            transform for predictions
        Useful when target is skewed.
        
    feval : callable, default None
        Score function (evaluation metric) which is used to calculate 
        results of cross-validation.
        If None, then by default:
            for regression - mean_absolute_error,
            for classification - accuracy_score
        You can use any function or define your own function like shown below:
        
        def mae(y_true, y_pred):
            return numpy.mean(numpy.abs(y_true - y_pred))
        
    n_folds : int, default 4
        Number of folds in cross-validation
        
    stratified : boolean, default False, meaningful only for classification task
        If True - use stratified folds in cross-validation
        
    shuffle : boolean, default False
        Wether to perform a shuffle before cross-validation split
        
    random_state : int, default 0
        Random seed for shuffle
        
    verbose : int, default 0
        Level of verbosity.
        0 - show no messages,
        1 - show single score for each 1-st level model,
        2 - show score for each fold of each 1-st level model
        
    Returns
    -------
    S_train : numpy array of shape [n_train_samples, n_models]
        Stacking features for train set
        
    S_test : numpy array of shape [n_test_samples, n_models]
        Stacking features for test set
    
    Usage
    -----
    # For regression
    S_train, S_test = stacking(models, X_train, y_train, X_test, verbose = 2)
    
    # For classification
    S_train, S_test = stacking(models, X_train, y_train, X_test, 
        regression = False, verbose = 2)
    
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
    
    Examples
    --------
    
    """
    # Whether to use log1p/expm1 transformations
    if log_transform and regression:
        do = True
    else:
        do = False

    # Specify default score function for cross-validation
    if feval is None and regression:
        feval = mean_absolute_error
    elif feval is None and not regression:
        feval = accuracy_score
        
    # Split indices to get folds (stratified is possible only for classification)
    if stratified and not regression:
        kf = StratifiedKFold(y_train, n_folds, shuffle = shuffle, random_state = random_state)
    else:
        kf = KFold(len(y_train), n_folds, shuffle = shuffle, random_state = random_state)

    # Create empty numpy arrays for stacking features
    S_train = np.zeros((len(X_train), len(models)))
    S_test = np.zeros((len(X_test), len(models)))
    
    # Loop across models
    for model_counter, model in enumerate(models):
        if verbose > 0:
            print('[model %d: %s] [eval metric: %s]' % (model_counter, model.__class__.__name__, feval.func_name))
            
        # Create empty numpy array, which will contain temporary predictions for test set made in each fold
        S_test_temp = np.zeros((len(X_test), len(kf)))
        
        # Loop across folds
        for fold_counter, (tr_index, te_index) in enumerate(kf):
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            
            # Fit 1-st level model
            model = model.fit(X_tr, log1p(y_tr, do = do))
            # Predict out-of-fold part of train set
            S_train[te_index, model_counter] = expm1(model.predict(X_te), do = do)
            # Predict full test set
            S_test_temp[:, fold_counter] = expm1(model.predict(X_test), do = do)
            
            if verbose > 1:
                print('    fold %d: [%.8f]' % (fold_counter, feval(y_te, S_train[te_index, model_counter])))
                
        # Compute mean or mode of predictions for test set
        if regression:
            S_test[:, model_counter] = np.mean(S_test_temp, axis = 1)
        else:
            S_test[:, model_counter] = st.mode(S_test_temp, axis = 1)[0].ravel()
            
        if verbose > 0:
            print('    ----')
            print('    TOTAL:  [%.8f]\n' % (feval(y_train, S_train[:, model_counter])))

    return (S_train, S_test)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

