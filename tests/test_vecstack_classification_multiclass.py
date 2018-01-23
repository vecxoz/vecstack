#-------------------------------------------------------------------------------
# !!! cross_val_predict uses stratified split
#-------------------------------------------------------------------------------
# Main concept for testing returned arrays:
# 1). create ground truth e.g. with cross_val_predict
# 2). run vecstack
# 3). compare returned arrays with ground truth 
# 4). compare arrays from file with ground truth 
#-------------------------------------------------------------------------------

import unittest
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

import os
import glob
import numpy as np
import scipy.stats as st
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from vecstack import stacking

n_classes = 3
n_folds = 5

X, y = make_classification(n_samples = 500, n_features = 5, n_informative = 3, n_redundant = 1, 
                           n_classes = n_classes, flip_y = 0, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class TestClassificationMulticlass(unittest.TestCase):

    def tearDown(self):
        # Remove files after each test
        files = glob.glob('*.npy')
        files.extend(glob.glob('*.txt'))
        for file in files:
            os.remove(file)
            
    #---------------------------------------------------------------------------
    # Test returned and saved arrays in each mode (parameter <mode>)
    # Here we also test parameter <stratified> 
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # Predict labels
    #---------------------------------------------------------------------------

    def test_oof_pred_mode(self):

        model = LogisticRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        models = [LogisticRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir = '.', 
            mode = 'oof_pred', random_state = 0, verbose = 0, stratified = True)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.npy'))[-1] # take the latest file
        S = np.load(file_name)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    def test_oof_mode(self):

        model = LogisticRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        S_test_1 = None

        models = [LogisticRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir = '.', 
            mode = 'oof', random_state = 0, verbose = 0, stratified = True)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.npy'))[-1] # take the latest file
        S = np.load(file_name)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    def test_pred_mode(self):

        model = LogisticRegression()
        S_train_1 = None
        _ = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        models = [LogisticRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir = '.', 
            mode = 'pred', random_state = 0, verbose = 0, stratified = True)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.npy'))[-1] # take the latest file
        S = np.load(file_name)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    def test_oof_pred_bag_mode(self):
        
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        # Usind StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = 0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = LogisticRegression()
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1 = st.mode(S_test_temp, axis = 1)[0]
    
        model = LogisticRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)

        models = [LogisticRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir = '.',
            mode = 'oof_pred_bag', random_state = 0, verbose = 0, stratified = True)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.npy'))[-1] # take the latest file
        S = np.load(file_name)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
    
    def test_pred_bag_mode(self):
        
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        # Usind StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = 0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = LogisticRegression()
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1 = st.mode(S_test_temp, axis = 1)[0]
    
        S_train_1 = None

        models = [LogisticRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir = '.',
            mode = 'pred_bag', random_state = 0, verbose = 0, stratified = True)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.npy'))[-1] # take the latest file
        S = np.load(file_name)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    #---------------------------------------------------------------------------
    # Predict proba
    #---------------------------------------------------------------------------
        
    def test_oof_pred_mode_proba(self):

        model = LogisticRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict_proba')
        _ = model.fit(X_train, y_train)
        S_test_1 = model.predict_proba(X_test)

        models = [LogisticRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, stratified = True,
            mode = 'oof_pred', random_state = 0, verbose = 0, needs_proba = True, save_dir = '.')
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.npy'))[-1] # take the latest file
        S = np.load(file_name)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    def test_oof_mode_proba(self):

        model = LogisticRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict_proba')
        S_test_1 = None

        models = [LogisticRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, stratified = True, 
            mode = 'oof', random_state = 0, verbose = 0, needs_proba = True, save_dir = '.')
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.npy'))[-1] # take the latest file
        S = np.load(file_name)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    def test_pred_mode_proba(self):

        model = LogisticRegression()
        S_train_1 = None
        _ = model.fit(X_train, y_train)
        S_test_1 = model.predict_proba(X_test)

        models = [LogisticRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, stratified = True, 
            mode = 'pred', random_state = 0, verbose = 0, needs_proba = True, save_dir = '.')
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.npy'))[-1] # take the latest file
        S = np.load(file_name)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    def test_oof_pred_bag_mode_proba(self):
        
        S_test_1 = np.zeros((X_test.shape[0], n_classes))
        S_test_temp = np.zeros((X_test.shape[0], n_folds * n_classes))
        # Using StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = 0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = LogisticRegression()
            _ = model.fit(X_tr, y_tr)
            col_slice_fold = slice(fold_counter * n_classes, fold_counter * n_classes + n_classes)
            S_test_temp[:, col_slice_fold] = model.predict_proba(X_test)
        for class_id in range(n_classes):
            S_test_1[:, class_id] = np.mean(S_test_temp[:, class_id::n_classes], axis = 1)
    
        model = LogisticRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict_proba')

        models = [LogisticRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir = '.',
            mode = 'oof_pred_bag', random_state = 0, verbose = 0, stratified = True, needs_proba = True)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.npy'))[-1] # take the latest file
        S = np.load(file_name)
        S_train_3 = S[0]
        S_test_3 = S[1]
        
        #@@@@
        # Look at proba
        # print('\nOne model')
        # print('etalon')
        # print(S_test_1[:2])
        # print('vecstack')
        # print(S_test_2[:2])
        #@@@@

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
    
    def test_pred_bag_mode_proba(self):
        
        S_test_1 = np.zeros((X_test.shape[0], n_classes))
        S_test_temp = np.zeros((X_test.shape[0], n_folds * n_classes))
        # Using StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = 0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = LogisticRegression()
            _ = model.fit(X_tr, y_tr)
            col_slice_fold = slice(fold_counter * n_classes, fold_counter * n_classes + n_classes)
            S_test_temp[:, col_slice_fold] = model.predict_proba(X_test)
        for class_id in range(n_classes):
            S_test_1[:, class_id] = np.mean(S_test_temp[:, class_id::n_classes], axis = 1)
    
        S_train_1 = None

        models = [LogisticRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir = '.',
            mode = 'pred_bag', random_state = 0, verbose = 0, stratified = True, needs_proba = True)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.npy'))[-1] # take the latest file
        S = np.load(file_name)
        S_train_3 = S[0]
        S_test_3 = S[1]
    
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    #---------------------------------------------------------------------------
    # Test <shuffle> and <random_state> parameters
    #---------------------------------------------------------------------------
    
    def test_oof_pred_bag_mode_shuffle(self):
        
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        # Usind StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = LogisticRegression()
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1 = st.mode(S_test_temp, axis = 1)[0]
    
        model = LogisticRegression()
        # !!! Important. Here we pass CV-generator not number of folds <cv = kf>
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = kf, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)

        models = [LogisticRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = True, save_dir = '.',
            mode = 'oof_pred_bag', random_state = 0, verbose = 0, stratified = True)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.npy'))[-1] # take the latest file
        S = np.load(file_name)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    #---------------------------------------------------------------------------
    # Test <metric> parameter and its default values depending on <regression> parameter
    # Labels
    # Important. We use <greater_is_better = True> in <make_scorer> for any error function
    # because we need raw scores (without minus sign)
    #---------------------------------------------------------------------------
    def test_oof_mode_metric(self):

        model = LogisticRegression()
        scorer = make_scorer(accuracy_score)
        scores = cross_val_score(model, X_train, y = y_train, cv = n_folds, 
            scoring = scorer, n_jobs = 1, verbose = 0)
        mean_str_1 = '%.8f' % np.mean(scores)
        std_str_1 = '%.8f' % np.std(scores)
        

        models = [LogisticRegression()]
        S_train, S_test = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, save_dir = '.', 
            mode = 'oof', random_state = 0, verbose = 0, stratified = True)
            
        # Load mean score and std from file
        # Normally if cleaning is performed there is only one .log.txt file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.log.txt'))[-1] # take the latest file
        with open(file_name) as f:
            for line in f:
                if 'MEAN' in line:
                    split = line.strip().split()
                    break

        mean_str_2 = split[1][1:-1]
        std_str_2 = split[3][1:-1]

        assert_equal(mean_str_1, mean_str_2)
        assert_equal(std_str_1, std_str_2)
    
    #---------------------------------------------------------------------------
    # Test <metric> parameter and its default values depending on <regression> parameter
    # Proba
    # Important. We use <greater_is_better = True> in <make_scorer> for any error function
    # because we need raw scores (without minus sign)
    #---------------------------------------------------------------------------
    def test_oof_mode_metric_proba(self):

        model = LogisticRegression()
        scorer = make_scorer(log_loss, needs_proba = True)
        scores = cross_val_score(model, X_train, y = y_train, cv = n_folds, 
            scoring = scorer, n_jobs = 1, verbose = 0)
        mean_str_1 = '%.8f' % np.mean(scores)
        std_str_1 = '%.8f' % np.std(scores)
        

        models = [LogisticRegression()]
        S_train, S_test = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, save_dir = '.', 
            mode = 'oof', random_state = 0, verbose = 0, stratified = True, 
            needs_proba = True)
            
        # Load mean score and std from file
        # Normally if cleaning is performed there is only one .log.txt file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.log.txt'))[-1] # take the latest file
        with open(file_name) as f:
            for line in f:
                if 'MEAN' in line:
                    split = line.strip().split()
                    break

        mean_str_2 = split[1][1:-1]
        std_str_2 = split[3][1:-1]

        assert_equal(mean_str_1, mean_str_2)
        assert_equal(std_str_1, std_str_2)
        
    #-------------------------------------------------------------------------------
    # Test several mdels in one run
    #-------------------------------------------------------------------------------
    
    def test_oof_pred_mode_2_models(self):

        # Model a
        model = LogisticRegression()
        S_train_1_a = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(X_train, y_train)
        S_test_1_a = model.predict(X_test).reshape(-1, 1)
        
        # Model b
        model = GaussianNB()
        S_train_1_b = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(X_train, y_train)
        S_test_1_b = model.predict(X_test).reshape(-1, 1)
        
        S_train_1 = np.c_[S_train_1_a, S_train_1_b]
        S_test_1 = np.c_[S_test_1_a, S_test_1_b]

        models = [LogisticRegression(),
                  GaussianNB()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir = '.', 
            mode = 'oof_pred', random_state = 0, verbose = 0, stratified = True)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.npy'))[-1] # take the latest file
        S = np.load(file_name)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    def test_oof_pred_bag_mode_2_models(self):
        
        # Model a
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        # Usind StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = 0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = LogisticRegression()
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1_a = st.mode(S_test_temp, axis = 1)[0]
    
        model = LogisticRegression()
        S_train_1_a = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
            
        # Model b
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        # Usind StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = 0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = GaussianNB()
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1_b = st.mode(S_test_temp, axis = 1)[0]
    
        model = GaussianNB()
        S_train_1_b = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
            
        S_train_1 = np.c_[S_train_1_a, S_train_1_b]
        S_test_1 = np.c_[S_test_1_a, S_test_1_b]

        models = [LogisticRegression(),
                  GaussianNB()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir = '.',
            mode = 'oof_pred_bag', random_state = 0, verbose = 0, stratified = True)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.npy'))[-1] # take the latest file
        S = np.load(file_name)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
        
    def test_oof_pred_mode_proba_2_models(self):

        # Model a
        model = LogisticRegression()
        S_train_1_a = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict_proba')
        _ = model.fit(X_train, y_train)
        S_test_1_a = model.predict_proba(X_test)
        
        # Model b
        model = GaussianNB()
        S_train_1_b = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict_proba')
        _ = model.fit(X_train, y_train)
        S_test_1_b = model.predict_proba(X_test)
        
        S_train_1 = np.c_[S_train_1_a, S_train_1_b]
        S_test_1 = np.c_[S_test_1_a, S_test_1_b]

        models = [LogisticRegression(),
                  GaussianNB()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, stratified = True,
            mode = 'oof_pred', random_state = 0, verbose = 0, needs_proba = True, save_dir = '.')
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.npy'))[-1] # take the latest file
        S = np.load(file_name)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
        
    def test_oof_pred_bag_mode_proba_2_models(self):
        
        # Model a
        S_test_1_a = np.zeros((X_test.shape[0], n_classes))
        S_test_temp = np.zeros((X_test.shape[0], n_folds * n_classes))
        # Using StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = 0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = LogisticRegression()
            _ = model.fit(X_tr, y_tr)
            col_slice_fold = slice(fold_counter * n_classes, fold_counter * n_classes + n_classes)
            S_test_temp[:, col_slice_fold] = model.predict_proba(X_test)
        for class_id in range(n_classes):
            S_test_1_a[:, class_id] = np.mean(S_test_temp[:, class_id::n_classes], axis = 1)
    
        model = LogisticRegression()
        S_train_1_a = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict_proba')
            
        # Model b
        S_test_1_b = np.zeros((X_test.shape[0], n_classes))
        S_test_temp = np.zeros((X_test.shape[0], n_folds * n_classes))
        # Using StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = 0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = GaussianNB()
            _ = model.fit(X_tr, y_tr)
            col_slice_fold = slice(fold_counter * n_classes, fold_counter * n_classes + n_classes)
            S_test_temp[:, col_slice_fold] = model.predict_proba(X_test)
        for class_id in range(n_classes):
            S_test_1_b[:, class_id] = np.mean(S_test_temp[:, class_id::n_classes], axis = 1)
    
        model = GaussianNB()
        S_train_1_b = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict_proba')
            
        S_train_1 = np.c_[S_train_1_a, S_train_1_b]
        S_test_1 = np.c_[S_test_1_a, S_test_1_b]
        
        

        models = [LogisticRegression(),
                  GaussianNB()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir = '.',
            mode = 'oof_pred_bag', random_state = 0, verbose = 0, stratified = True, needs_proba = True)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob('*.npy'))[-1] # take the latest file
        S = np.load(file_name)
        S_train_3 = S[0]
        S_test_3 = S[1]
        
        #@@@@
        # Look at proba
        # print('\nTwo models')
        # print('etalon')
        # print(S_test_1[:2])
        # print('vecstack')
        # print(S_test_2[:2])
        #@@@@

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
    
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

