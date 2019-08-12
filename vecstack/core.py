"""Functional API for stacking.

Find out how to use:
>>> from vecstack import stacking
>>> help(stacking)

MIT License

Copyright (c) 2016-2018 Igor Ivanov
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

from __future__ import print_function
from __future__ import division

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

import os
import sys
import warnings
from datetime import datetime
import numpy as np
import scipy.stats as st
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.base import clone

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

def model_action(model, X_train, y_train, X_test, 
                 sample_weight=None, action=None, 
                 transform=None):
    """Performs model action.
    This wrapper gives us ability to choose action dynamically 
    (e.g. predict or predict_proba).
    Note. Through <model_action> and then through <transformer> we
    apply <transform_target> and <transform_pred> functions if given by user
    on the target and prediction in each fold separately 
    to be able to calculate proper scores
    """
    if 'fit' == action:
        # We use following condition, because some models (e.g. Lars) may not have
        # 'sample_weight' parameter of fit method
        if sample_weight is not None:
            return model.fit(X_train, transformer(y_train, func = transform), sample_weight=sample_weight)
        else:
            return model.fit(X_train, transformer(y_train, func = transform))
    elif 'predict' == action:
        return transformer(model.predict(X_test), func = transform)
    elif 'predict_proba' == action:
        return transformer(model.predict_proba(X_test), func = transform)
    else:
        raise ValueError('Parameter action must be set properly')

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def model_params(model):
    """
    Create string of alphabetically sorted parameters of the model
    obtained with get_params or string contaning result of __repr__ call
    """
    s = ''
    
    if hasattr(model, 'get_params'):
        params_dict = model.get_params()
        max_len = 0
        for key in params_dict:
            if len(key) > max_len:
                max_len = len(key)
        sorted_keys = sorted(params_dict.keys())
        for key in sorted_keys:
            s += '%-*s %s\n' % (max_len, key, params_dict[key])
            
    elif hasattr(model, '__repr__'):
        s = model.__repr__()
        s += '\n'
    
    else:
        s = 'Model has no ability to show parameters (has no <get_params> or <__repr__>)\n'
        
    s += '\n'
        
    return s
    
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def stacking(models, X_train, y_train, X_test, 
             sample_weight=None, regression=True,
             transform_target=None, transform_pred=None,
             mode='oof_pred_bag', needs_proba=False, save_dir=None,
             metric=None, n_folds=4, stratified=False,
             shuffle=False, random_state=0, verbose=0):
    """Function 'stacking' takes train data, test data and list of 1-st level
    models, and returns stacking features, which can be used with 2-nd level model.
    
    Complete examples and stacking concept - see below.
    
    Parameters
    ----------
    models : list 
        List of 1-st level models. You can use any models that follow sklearn
        convention i.e. accept numpy arrays and have 
        methods 'fit', 'predict' and 'predict_proba'.
        Following sklearn convention in binary classification 
        task with probabilities model must return probabilities 
        for each class (i.e. two columns).
        
    X_train : numpy array or sparse matrix of N-dim shape, e.g. 2-dim [n_train_samples, n_features]
        Training data
    
    y_train : numpy 1d array
        Target values
        
    X_test : numpy array or sparse matrix of N-dim shape, e.g. 2-dim [n_test_samples, n_features]
        Test data
        
    sample_weight : numpy array of shape [n_train_samples]
        Individual weights for each sample (passed to fit method of the model).
        Note: sample_weight has length of full training set X_train and it would be
        split automatically for each fold.
        
    regression : boolean, default True
        If True - perform stacking for regression task, 
        if False - perform stacking for classification task
        
    transform_target : callable, default None
        Function to transform target variable.
        If None - transformation is not used.
        For example, for regression task (if target variable is skewed)
            you can use transformation like numpy.log1p.
            Set transform_target = numpy.log1p
        Usually you want to use respective backward transformation 
            for prediction like numpy.expm1.
            Set transform_pred = numpy.expm1
        Caution! Some transformations may give inapplicable results. 
            For example, if target variable contains zeros, numpy.log 
            gives you -inf. In such case you can use appropriate 
            transformation like numpy.log1p and respective
            backward transformation like numpy.expm1
        
    transform_pred : callable, default None
        Function to transform prediction.
        If None - transformation is not used.
        If you use transformation for target variable (transform_target)
            like numpy.log1p, then using transform_pred you can specify 
            respective backward transformation like numpy.expm1.
        Look at description of parameter transform_target
        
    mode: str, default 'oof_pred_bag' (alias 'A')
        Note: for detailes see terminology below
        'oof' - return only oof
        'oof_pred' (alias 'B') - return oof and pred
        'oof_pred_bag' (alias 'A') - return oof and bagged pred
        'pred' - return pred only
        'pred_bag' - return bagged pred only
        Terminology:
            oof - out-of-fold predictions for train set
            pred - predictions for tests set (model is fitted once on full train set, then predicts test set)
            bagged pred - bagged predictions for tests set (given that we have N folds, 
                we fit N models on each fold's train data, then each model predicts test set,
                then we perform bagging: compute mean of predicted values (for regression or class probabilities) 
                or majority voting: compute mode (when predictions are class labels)
        
    needs_proba: boolean, default False, meaningful only for classification task
        Whether to predict probabilities (instead of class labels)
        in classification task.
        Ignored if regression=True
        
    save_dir: str, default None
        If specified - considered as a valid directory (must exist) where log and 
        returned arrays will be saved. 
        If not specified - log and arrays will not be saved.
        Path may be absolute or relative to the directory from where script was run.
        Absolute examples: Win: 'c:/some/dir', Linux: '/home/user/run'
        Relative examples: Win and Linux current directory: '.'
        Both arrays are saved in a single .npy file, so you can load it as follows:
        S = np.load('c:/some/dir/[2017.11.29].[13.47.31].250824.45dc2b.npy')
        S_train = S[0]
        S_test = S[1]
        Log is saved in plain text.
        File names are the current timestamp plus random part to ensure uniqueness.
        
    metric : callable, default None
        Evaluation metric (score function) which is used to calculate 
        results of cross-validation.
        If None, then by default:
            sklearn.metrics.mean_absolute_error - for regression
            sklearn.metrics.accuracy_score - for classification with class labels
            sklearn.metrics.log_loss - for classification with probabilities
        You can use any appropriate sklearn metric or 
            define your own metric like shown below:
        
        def your_metric(y_true, y_pred):
            # calculate
            return result
            
        MEAN/FULL interpretation:
            MEAN - mean (average) of scores for each fold.
            FULL - metric calculated using combined oof predictions
                for full train set and target.
            For some metrics (e.g. rmse, rmsle) MEAN and FULL values are 
                slightly different
        
    n_folds : int, default 4
        Number of folds in cross-validation
        
    stratified : boolean, default False, meaningful only for classification task
        If True - use stratified folds in cross-validation
        Ignored if regression=True
        
    shuffle : boolean, default False
        Whether to perform a shuffle before cross-validation split
        
    random_state : int, default 0
        Random seed
        
    verbose : int, default 0
        Level of verbosity.
        0 - show no messages
        1 - for each model show mean score
        2 - for each model show score for each fold and mean score
        
    Returns
    -------
    S_train : numpy array of shape [n_train_samples, n_models] or None
        Stacking features for train set
        
    S_test : numpy array of shape [n_test_samples, n_models] or None
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
    # Initialize 1-st level models.
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

    # Initialize 2-nd level model
    model = XGBRegressor(seed = 0, n_jobs = -1, learning_rate = 0.1, 
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
    # Initialize 1-st level models.
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

    # Initialize 2-nd level model
    model = XGBClassifier(seed = 0, n_jobs = -1, learning_rate = 0.1, 
        n_estimators = 100, max_depth = 3)
    
    # Fit 2-nd level model
    model = model.fit(S_train, y_train)

    # Predict
    y_pred = model.predict(S_test)

    # Final prediction score
    print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred))
    """
    #---------------------------------------------------------------------------
    # Check parameters
    #---------------------------------------------------------------------------
    # If empty <models> list
    if 0 == len(models):
        raise ValueError('List of models is empty')
    # Check arrays
    # y_train and sample_weight must be 1d ndarrays (i.e. row, not column)
    X_train, y_train = check_X_y(X_train,
                                 y_train,
                                 accept_sparse=['csr'], # allow csr and cast all other sparse types to csr
                                 force_all_finite=False, # allow nan and inf because 
                                                         # some models (xgboost) can handle
                                 allow_nd=True,
                                 multi_output=False) # do not allow several columns in y_train
                                 
    if X_test is not None: # allow X_test to be None for mode='oof'
        X_test = check_array(X_test,
                             accept_sparse=['csr'], # allow csr and cast all other sparse types to csr
                             allow_nd=True,
                             force_all_finite=False) # allow nan and inf because 
                                                     # some models (xgboost) can handle
    if sample_weight is not None:
        sample_weight = np.array(sample_weight).ravel()
    # <regression>
    regression = bool(regression)
    # If wrong <mode>
    if mode not in ['pred', 'pred_bag', 'oof', 'oof_pred', 'B', 'oof_pred_bag', 'A']:
        raise ValueError('Parameter <mode> must be set properly')
    # <needs_proba>
    needs_proba = bool(needs_proba)
    # If wrong <save_dir>
    if save_dir is not None:
        save_dir = os.path.normpath(save_dir)
        if not os.path.isdir(save_dir):
            raise ValueError('Path does not exist or is not a directory. Check <save_dir> parameter')
    # <n_folds>
    if not isinstance(n_folds, int):
        raise ValueError('Parameter <n_folds> must be integer')
    if not n_folds > 1:
        raise ValueError('Parameter <n_folds> must be not less than 2')
    # <stratified>
    stratified = bool(stratified)
    # <shuffle>
    shuffle = bool(shuffle)
    # <verbose>
    if verbose not in [0, 1, 2]:
        raise ValueError('Parameter <verbose> must be 0, 1, or 2')
    # Additional check for inapplicable parameter combinations
    # If regression=True we ignore classification-specific parameters and issue user warning
    if regression and (needs_proba or stratified):
        warn_str = 'This is regression task hence classification-specific parameters set to <True> were ignored:'
        if needs_proba:
            needs_proba = False
            warn_str += ' <needs_proba>'
        if stratified:
            stratified = False
            warn_str += ' <stratified>'
        warnings.warn(warn_str, UserWarning)
    #---------------------------------------------------------------------------
    # Specify default metric
    #---------------------------------------------------------------------------
    if metric is None and regression:
        metric = mean_absolute_error
    elif metric is None and not regression:
        if needs_proba:
            metric = log_loss
        else:
            metric = accuracy_score
    #---------------------------------------------------------------------------
    # Create report header strings and print report header
    #---------------------------------------------------------------------------
    if save_dir is not None or verbose > 0:
        if regression:
            task_str = 'task:         [regression]'
        else:
            task_str = 'task:         [classification]'
            n_classes_str = 'n_classes:    [%d]' % len(np.unique(y_train))
        metric_str = 'metric:       [%s]' % metric.__name__
        mode_str = 'mode:         [%s]' % mode
        n_models_str = 'n_models:     [%d]' % len(models)
    
    # Print report header
    if verbose > 0:
        print(task_str)
        if not regression:
            print(n_classes_str)
        print(metric_str)
        print(mode_str)
        print(n_models_str + '\n')
    #---------------------------------------------------------------------------
    # Split indices to get folds (stratified can be used only for classification)
    #---------------------------------------------------------------------------
    if not regression and stratified:
        kf = StratifiedKFold(n_splits = n_folds, shuffle = shuffle, random_state = random_state)
    else:
        kf = KFold(n_splits = n_folds, shuffle = shuffle, random_state = random_state)
    #---------------------------------------------------------------------------
    # Compute number of classes (if we need probabilities) to create appropreate empty arrays
    # !!! Important. In order to unify array creation variable <n_classes> is always
    # equal to 1, except the case when we performing classification task with needs_proba=True
    #---------------------------------------------------------------------------
    if not regression and needs_proba:
        n_classes = len(np.unique(y_train))
        action = 'predict_proba'
    else:
        n_classes = 1
        action = 'predict'
    #---------------------------------------------------------------------------
    # Create empty numpy arrays for OOF
    #---------------------------------------------------------------------------
    if mode in ['oof_pred', 'B', 'oof_pred_bag', 'A']:
        S_train = np.zeros(( X_train.shape[0], len(models) * n_classes ))
        S_test = np.zeros(( X_test.shape[0], len(models) * n_classes ))
    elif mode in ['oof']:
        S_train = np.zeros(( X_train.shape[0], len(models) * n_classes ))
        S_test = None
    elif mode in ['pred', 'pred_bag']:
        S_train = None
        S_test = np.zeros(( X_test.shape[0], len(models) * n_classes ))

    #---------------------------------------------------------------------------
    # High-level function variables
    #---------------------------------------------------------------------------
    # String to store models-folds part of log
    models_folds_str = ''
    
    #---------------------------------------------------------------------------
    # Loop across models
    #---------------------------------------------------------------------------
    for model_counter, model in enumerate(models):
        if save_dir is not None or verbose > 0:
            model_str = 'model %2d:     [%s]' % (model_counter, model.__class__.__name__)
        if save_dir is not None:
            models_folds_str += '-' * 40 + '\n'
            models_folds_str += model_str + '\n'
            models_folds_str += '-' * 40 + '\n\n'
            models_folds_str += model_params(model)
        if verbose > 0:
            print(model_str)
            
        # Create empty numpy array, which will contain temporary predictions for test set made in each fold
        if mode in ['pred_bag', 'oof_pred_bag', 'A']:
            S_test_temp = np.zeros((X_test.shape[0], n_folds * n_classes))
        
        # Create empty array to store scores for each fold (to find mean)
        scores = np.array([])
        
        #-----------------------------------------------------------------------
        # Loop across folds
        #-----------------------------------------------------------------------
        if mode in ['pred_bag', 'oof', 'oof_pred', 'B', 'oof_pred_bag', 'A']:
            for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
                # Split data and target
                X_tr = X_train[tr_index]
                y_tr = y_train[tr_index]
                X_te = X_train[te_index]
                y_te = y_train[te_index]
                
                # Split sample weights accordingly (if passed)
                if sample_weight is not None:
                    sample_weight_tr = sample_weight[tr_index]
                    # sample_weight_te = sample_weight[te_index]
                else:
                    sample_weight_tr = None
                    # sample_weight_te = None

                # Save RAM: clone to avoid fitting model directly inside users list
                # Set safe=False to be able to clone non-sklearn models
                model = clone(model, safe=False)
                
                # Fit 1-st level model
                if mode in ['pred_bag', 'oof', 'oof_pred', 'B', 'oof_pred_bag', 'A']:
                    _ = model_action(model, X_tr, y_tr, None, sample_weight = sample_weight_tr, action = 'fit', transform = transform_target)
                    
                # Predict out-of-fold part of train set
                if mode in ['oof', 'oof_pred', 'B', 'oof_pred_bag', 'A']:
                    if 'predict_proba' == action:
                        col_slice_model = slice(model_counter * n_classes, model_counter * n_classes + n_classes)
                    else:
                        col_slice_model = model_counter
                    S_train[te_index, col_slice_model] = model_action(model, None, None, X_te, action = action, transform = transform_pred)
                    
                # Predict full test set in each fold
                if mode in ['pred_bag', 'oof_pred_bag', 'A']:
                    if 'predict_proba' == action:
                        col_slice_fold = slice(fold_counter * n_classes, fold_counter * n_classes + n_classes)
                    else:
                        col_slice_fold = fold_counter
                    S_test_temp[:, col_slice_fold] = model_action(model, None, None, X_test, action = action, transform = transform_pred)
                        
                # Compute scores
                if mode in ['oof', 'oof_pred', 'B', 'oof_pred_bag', 'A']:
                    if save_dir is not None or verbose > 0:
                        score = metric(y_te, S_train[te_index, col_slice_model])
                        scores = np.append(scores, score)
                        fold_str = '    fold %2d:  [%.8f]' % (fold_counter, score)
                    if save_dir is not None:
                        models_folds_str += fold_str + '\n'
                    if verbose > 1:    
                        print(fold_str)
                
        # Compute mean or mode of predictions for test set in bag modes
        if mode in ['pred_bag', 'oof_pred_bag', 'A']:
            if 'predict_proba' == action:
                # Here we copute means of probabilirties for each class
                for class_id in range(n_classes):
                    S_test[:, model_counter * n_classes + class_id] = np.mean(S_test_temp[:, class_id::n_classes], axis = 1)
            else:
                if regression:
                    S_test[:, model_counter] = np.mean(S_test_temp, axis = 1)
                else:
                    S_test[:, model_counter] = st.mode(S_test_temp, axis = 1)[0].ravel()
            
        # Compute scores: mean + std and full
        if mode in ['oof', 'oof_pred', 'B', 'oof_pred_bag', 'A']:
            if save_dir is not None or verbose > 0:
                sep_str = '    ----'
                mean_str = '    MEAN:     [%.8f] + [%.8f]' % (np.mean(scores), np.std(scores))
                full_str = '    FULL:     [%.8f]\n' % (metric(y_train, S_train[:, col_slice_model]))
            if save_dir is not None:
                models_folds_str += sep_str + '\n'
                models_folds_str += mean_str + '\n'
                models_folds_str += full_str + '\n'
            if verbose > 0:
                print(sep_str)
                print(mean_str)
                print(full_str)
                
        # Fit model on full train set and predict test set
        if mode in ['pred', 'oof_pred', 'B']:
            if verbose > 0:
                print('    Fitting on full train set...\n')
            _ = model_action(model, X_train, y_train, None, sample_weight = sample_weight, action = 'fit', transform = transform_target)
            if 'predict_proba' == action:
                col_slice_model = slice(model_counter * n_classes, model_counter * n_classes + n_classes)
            else:
                col_slice_model = model_counter
            S_test[:, col_slice_model] = model_action(model, None, None, X_test, action = action, transform = transform_pred)
    #---------------------------------------------------------------------------
    # Cast class labels to int
    #---------------------------------------------------------------------------
    if not regression and not needs_proba:
        if S_train is not None:
            S_train = S_train.astype(int)
        if S_test is not None:
            S_test = S_test.astype(int)
    #---------------------------------------------------------------------------
    # Save OOF and log
    #---------------------------------------------------------------------------
    if save_dir is not None:
        try:
            # We have already done save_dir = os.path.normpath(save_dir)
            
            # We generate random last 6 symbols to ensure that name is unique
            time_str = datetime.now().strftime('[%Y.%m.%d].[%H.%M.%S].%f') + ('.%06x' % np.random.randint(0xffffff))
            
            # Prepare paths for OFF and log files
            file_name = time_str + '.npy'
            log_file_name = time_str + '.log.txt'
            
            full_path = os.path.join(save_dir, file_name)
            log_full_path = os.path.join(save_dir, log_file_name)
            
            # Save OOF
            np.save(full_path, np.array([S_train, S_test]))
            
            # Save log
            log_str = 'vecstack log '
            log_str += time_str + '\n\n'
            log_str += task_str + '\n'
            if not regression:
                log_str += n_classes_str + '\n'
            log_str += metric_str + '\n'
            log_str += mode_str + '\n'
            log_str += n_models_str + '\n\n'
            log_str += models_folds_str
            log_str += '-' * 40 + '\n'
            log_str += 'END\n'
            log_str += '-' * 40 + '\n'
            
            with open(log_full_path, 'w') as f:
                _ = f.write(log_str)

            if verbose > 0:
                print('Result was saved to [%s]' % full_path)
        except:
            print('Error while saving files:\n%s' % sys.exc_info()[1])

    #---------------------------------------------------------------------------
    # Return
    #---------------------------------------------------------------------------
    # For consistency we always return two values: 
    # 1-st - for train, 2-nd - for test
    # Some of this values may be None
    return (S_train, S_test)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

