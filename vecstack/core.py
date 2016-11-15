"""Python package for stacking.
Author: <vecxoz@gmail.com>
"""

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

from numpy import zeros
from numpy import mean
from scipy.stats import mode
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def stacking(models, X_train, y_train, X_test, regression = True, 
    stratified = False, n_folds = 4, feval = None, shuffle = False, 
    random_state = 0, verbose = 0):
    """Function [stacking] takes train data, test data and list of 1-st level
    models, and return stacking features, which can be used with 2-nd level model.
    
    Complete examples and stacking concept - see below.
    
    Parameters
    ----------
    models : list 
        List of 1-st level models. You can use any models that follow sklearn
        convention i.e. have methods [fit] and [predict].
        
    X_train : numpy array or sparse matrix of shape [n_train_samples, n_features]
        Training data
    
    y_train : numpy 1d array
        Target values
        
    X_test : numpy array or sparse matrix of shape [n_test_samples, n_features]
        Test data
        
    regression : boolean, default True
        If True - perfom stacking for regression task, 
        if False - perfom stacking for classification task
        
    stratified : boolean, default False, meaningful only for classification task
        If True - use stratified folds in cross-validation
        
    n_folds : int, default 4
        Number of folds in cross-validation
        
    feval : callable, default None
        Score function (evaluation metric) which is used to calculate 
        results of cross-validation.
        If None, then by default:
            for regression - mean_absolute_error,
            for classification - accuracy_score
        You can use any function or define your own function like shown below:
        
        def mae(y_true, y_pred):
            return numpy.mean(numpy.abs(y_true - y_pred))
        
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
    
    Examples
    --------
    
    Stacking concept
    ----------------
    
    """
    # Specify default score function for cross-validation
    if feval is None and regression:
        feval = mean_absolute_error
    elif feval is None and not regression:
        feval = accuracy_score
        
    # Split indices to get folds (stratified is possible only for classification)
    if stratified and not regression:
        kf = list(StratifiedKFold(y_train, n_folds, shuffle = shuffle, random_state = random_state))
    else:
        kf = list(KFold(len(y_train), n_folds, shuffle = shuffle, random_state = random_state))

    # Create empty numpy arrays for stacking features
    S_train = zeros((len(X_train), len(models)))
    S_test = zeros((len(X_test), len(models)))
    
    # Loop across models
    for model_counter, model in enumerate(models):
        if verbose > 0:
            print('[model %d: %s] [eval metric: %s]' % (model_counter, model.__class__.__name__, feval.func_name))
            
        # Create empty numpy array, wich will contain temporary predictions for test set made in each fold
        S_test_temp = zeros((len(X_test), len(kf)))
        
        # Loop across folds
        for fold_counter, (tr_index, te_index) in enumerate(kf):
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            
            # Fit 1-st level model
            model = model.fit(X_tr, y_tr)
            # Predict out-of-fold part of train set
            S_train[te_index, model_counter] = model.predict(X_te)
            # Predict full test set
            S_test_temp[:, fold_counter] = model.predict(X_test)
            
            if verbose > 1:
                print('    fold %d: [%.8f]' % (fold_counter, feval(y_te, S_train[te_index, model_counter])))
                
        # Compute mean or mode of predictions for test set
        if regression:
            S_test[:, model_counter] = mean(S_test_temp, axis = 1)
        else:
            S_test[:, model_counter] = mode(S_test_temp, axis = 1)[0].ravel()
            
        if verbose > 0:
            print('    ----')
            print('    TOTAL:  [%.8f]\n' % (feval(y_train, S_train[:, model_counter])))

    return (S_train, S_test)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

