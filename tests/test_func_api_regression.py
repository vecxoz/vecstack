#-------------------------------------------------------------------------------
# Main concept for testing returned arrays:
# 1). create ground truth e.g. with cross_val_predict
# 2). run vecstack
# 3). compare returned arrays with ground truth 
# 4). compare arrays from file with ground truth 
#-------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import division

import unittest
from numpy.testing import assert_array_equal
# from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from numpy.testing import assert_warns

import os
import glob
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# from sklearn.datasets import load_boston
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from vecstack import stacking
from vecstack.core import model_action

n_folds = 5
temp_dir = 'tmpdw35lg54ms80eb42'

# boston = load_boston()
boston = fetch_openml(name='boston', version=1, as_frame=False, parser='auto')
# X, y = boston.data, boston.target
X, y = boston.data.astype(float), boston.target.astype(float)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Make train/test split by hand to avoid strange errors probably related to testing suit:
# https://github.com/scikit-learn/scikit-learn/issues/1684
# https://github.com/scikit-learn/scikit-learn/issues/1704
# Note: Python 2.7, 3.4 - OK, but 3.5, 3.6 - error

np.random.seed(0)
ind = np.arange(500)
np.random.shuffle(ind)

ind_train = ind[:400]
ind_test = ind[400:]

X_train = X[ind_train]
X_test = X[ind_test]

y_train = y[ind_train]
y_test = y[ind_test]


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class MinimalEstimator:
    """Has no get_params attribute"""
    def __init__(self, random_state=0):
        self.random_state = random_state
    def __repr__(self):
        return 'Demo string from __repr__'
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.ones(X.shape[0])
    def predict_proba(self, X):
        return np.zeros(X.shape[0])

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class TestFuncRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            os.mkdir(temp_dir)
        except:
            print('Unable to create temp dir')
            
    @classmethod
    def tearDownClass(cls):
        try:
            os.rmdir(temp_dir)
        except:
            print('Unable to remove temp dir')
    
    def tearDown(self):
        # Remove files after each test
        files = glob.glob(os.path.join(temp_dir, '*.npy'))
        files.extend(glob.glob(os.path.join(temp_dir, '*.log.txt')))
        try:
            for file in files:
                os.remove(file)
        except:
            print('Unable to remove temp file')
    
    #---------------------------------------------------------------------------
    # Testing returned and saved arrays in each mode
    #---------------------------------------------------------------------------

    def test_oof_pred_mode(self):
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred', random_state = 0, verbose = 0)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)

    def test_B_mode(self):
        """ 'B' is alias for 'oof_pred' """
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds,
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test,
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'B', random_state = 0, verbose = 0)

        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)

        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)

    def test_oof_mode(self):

        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        S_test_1 = None

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir, 
            mode = 'oof', random_state = 0, verbose = 0)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    def test_pred_mode(self):
    
        model = LinearRegression()
        S_train_1 = None
        _ = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir, 
            mode = 'pred', random_state = 0, verbose = 0)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    def test_oof_pred_bag_mode(self):
        
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        kf = KFold(n_splits = n_folds, shuffle = False, random_state = None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = LinearRegression()
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1 = np.mean(S_test_temp, axis = 1).reshape(-1, 1)
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred_bag', random_state = 0, verbose = 0)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)

    def test_A_mode(self):
        """ 'A' is alias for 'oof_pred_bag' """
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        kf = KFold(n_splits = n_folds, shuffle = False, random_state = None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = LinearRegression()
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1 = np.mean(S_test_temp, axis = 1).reshape(-1, 1)

        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds,
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test,
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'A', random_state = 0, verbose = 0)

        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)

        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
    
    def test_pred_bag_mode(self):
        
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        kf = KFold(n_splits = n_folds, shuffle = False, random_state = None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = LinearRegression()
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1 = np.mean(S_test_temp, axis = 1).reshape(-1, 1)

        S_train_1 = None

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'pred_bag', random_state = 0, verbose = 0)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    #---------------------------------------------------------------------------
    # Testing <sample_weight> all ones
    #---------------------------------------------------------------------------
    def test_oof_pred_mode_sample_weight_one(self):
    
        sw = np.ones(len(y_train))
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict', 
            params = {'sample_weight': sw}).reshape(-1, 1)
        _ = model.fit(X_train, y_train, sample_weight = sw)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred', random_state = 0, verbose = 0,
            sample_weight = sw)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
    
    #---------------------------------------------------------------------------
    # Test <sample_weight> all random
    #---------------------------------------------------------------------------
    def test_oof_pred_mode_sample_weight_random(self):
    
        np.random.seed(0)
        sw = np.random.rand(len(y_train))
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict', 
            params = {'sample_weight': sw}).reshape(-1, 1)
        _ = model.fit(X_train, y_train, sample_weight = sw)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred', random_state = 0, verbose = 0,
            sample_weight = sw)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    #---------------------------------------------------------------------------
    # Testing <transform_target> and <transform_pred> parameters
    #---------------------------------------------------------------------------
    
    def test_oof_pred_mode_transformations(self):
    
        model = LinearRegression()
        S_train_1 = np.expm1(cross_val_predict(model, X_train, y = np.log1p(y_train), cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict')).reshape(-1, 1)
        _ = model.fit(X_train, np.log1p(y_train))
        S_test_1 = np.expm1(model.predict(X_test)).reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred', random_state = 0, verbose = 0,
            transform_target = np.log1p, transform_pred = np.expm1)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    #---------------------------------------------------------------------------
    # Testing <verbose> parameter
    #---------------------------------------------------------------------------
    def test_oof_pred_mode_verbose_1(self):
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)
        
        
        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir, 
            mode = 'oof_pred', random_state = 0, verbose = 0)

        models = [LinearRegression()]
        S_train_3, S_test_3 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir, 
            mode = 'oof_pred', random_state = 0, verbose = 1)
            
        models = [LinearRegression()]
        S_train_4, S_test_4 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir, 
            mode = 'oof_pred', random_state = 0, verbose = 2)
            
        models = [LinearRegression()]
        S_train_5, S_test_5 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, 
            mode = 'oof_pred', random_state = 0, verbose = 0)
            
        models = [LinearRegression()]
        S_train_6, S_test_6 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, 
            mode = 'oof_pred', random_state = 0, verbose = 1)
            
        models = [LinearRegression()]
        S_train_7, S_test_7 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, 
            mode = 'oof_pred', random_state = 0, verbose = 2)
            

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
        assert_array_equal(S_train_1, S_train_4)
        assert_array_equal(S_test_1, S_test_4)
        
        assert_array_equal(S_train_1, S_train_5)
        assert_array_equal(S_test_1, S_test_5)
        
        assert_array_equal(S_train_1, S_train_6)
        assert_array_equal(S_test_1, S_test_6)
        
        assert_array_equal(S_train_1, S_train_7)
        assert_array_equal(S_test_1, S_test_7)

    #---------------------------------------------------------------------------
    # Test <metric> parameter and its default values depending on <regression> parameter
    # Important. We use <greater_is_better = True> in <make_scorer> for any error function
    # because we need raw scores (without minus sign)
    #---------------------------------------------------------------------------
    def test_oof_mode_metric(self):

        model = LinearRegression()
        scorer = make_scorer(mean_absolute_error)
        scores = cross_val_score(model, X_train, y = y_train, cv = n_folds, 
            scoring = scorer, n_jobs = 1, verbose = 0)
        mean_str_1 = '%.8f' % np.mean(scores)
        std_str_1 = '%.8f' % np.std(scores)
        

        models = [LinearRegression()]
        S_train, S_test = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, save_dir=temp_dir, 
            mode = 'oof', random_state = 0, verbose = 0)
            
        # Load mean score and std from file
        # Normally if cleaning is performed there is only one .log.txt file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.log.txt')))[-1] # take the latest file
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
    
        model = LinearRegression()
        S_train_1_a = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(X_train, y_train)
        S_test_1_a = model.predict(X_test).reshape(-1, 1)
        
        model = Ridge(random_state = 0)
        S_train_1_b = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(X_train, y_train)
        S_test_1_b = model.predict(X_test).reshape(-1, 1)
        
        S_train_1 = np.c_[S_train_1_a, S_train_1_b]
        S_test_1 = np.c_[S_test_1_a, S_test_1_b]

        models = [LinearRegression(),
                  Ridge(random_state = 0)]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred', random_state = 0, verbose = 0)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    def test_oof_pred_bag_mode_2_models(self):
        
        # Model a
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        kf = KFold(n_splits = n_folds, shuffle = False, random_state = None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = LinearRegression()
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1_a = np.mean(S_test_temp, axis = 1).reshape(-1, 1)
    
        model = LinearRegression()
        S_train_1_a = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
            
        # Model b
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        kf = KFold(n_splits = n_folds, shuffle = False, random_state = None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = Ridge(random_state = 0)
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1_b = np.mean(S_test_temp, axis = 1).reshape(-1, 1)
    
        model = Ridge(random_state = 0)
        S_train_1_b = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
            
        S_train_1 = np.c_[S_train_1_a, S_train_1_b]
        S_test_1 = np.c_[S_test_1_a, S_test_1_b]
        

        models = [LinearRegression(),
                  Ridge(random_state = 0)]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred_bag', random_state = 0, verbose = 0)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    #---------------------------------------------------------------------------
    # Testing sparse types CSR, CSC, COO
    #---------------------------------------------------------------------------

    def test_oof_pred_mode_sparse_csr(self):
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, csr_matrix(X_train), y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(csr_matrix(X_train), y_train)
        S_test_1 = model.predict(csr_matrix(X_test)).reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, csr_matrix(X_train), y_train, csr_matrix(X_test), 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred', random_state = 0, verbose = 0)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    def test_oof_pred_mode_sparse_csc(self):
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, csc_matrix(X_train), y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(csc_matrix(X_train), y_train)
        S_test_1 = model.predict(csc_matrix(X_test)).reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, csc_matrix(X_train), y_train, csc_matrix(X_test),
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred', random_state = 0, verbose = 0)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    def test_oof_pred_mode_sparse_coo(self):
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, coo_matrix(X_train), y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(coo_matrix(X_train), y_train)
        S_test_1 = model.predict(coo_matrix(X_test)).reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, coo_matrix(X_train), y_train, coo_matrix(X_test),
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred', random_state = 0, verbose = 0)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    #---------------------------------------------------------------------------
    # Testing X_train -> SCR, X_test -> COO
    #---------------------------------------------------------------------------
    
    def test_oof_pred_mode_sparse_csr_coo(self):
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, csr_matrix(X_train), y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(csr_matrix(X_train), y_train)
        S_test_1 = model.predict(coo_matrix(X_test)).reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, csr_matrix(X_train), y_train, coo_matrix(X_test),
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred', random_state = 0, verbose = 0)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    #---------------------------------------------------------------------------
    # Testing X_train -> SCR, X_test -> Dense
    #---------------------------------------------------------------------------
    
    def test_oof_pred_mode_sparse_csr_dense(self):
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, csr_matrix(X_train), y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(csr_matrix(X_train), y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, csr_matrix(X_train), y_train, X_test,
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred', random_state = 0, verbose = 0)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
    
    #---------------------------------------------------------------------------
    # Testing X_test=None
    #---------------------------------------------------------------------------
    def test_oof_mode_xtest_is_none(self):

        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        S_test_1 = None

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, None, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir, 
            mode = 'oof', random_state = 0, verbose = 0)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    #---------------------------------------------------------------------------
    # Testing parameter exceptions
    #---------------------------------------------------------------------------
    def test_exceptions(self):
        # Empty model list
        assert_raises(ValueError, stacking, [], X_train, y_train, X_test)
        # Wrong mode
        assert_raises(ValueError, stacking, [LinearRegression()], 
                      X_train, y_train, X_test, mode='abc')
        # Path does not exist
        assert_raises(ValueError, stacking, [LinearRegression()], 
                      X_train, y_train, X_test, save_dir='./As26bV85')
        # n_folds is not int
        assert_raises(ValueError, stacking, [LinearRegression()], 
                      X_train, y_train, X_test, n_folds='A')
        # n_folds is less than 2
        assert_raises(ValueError, stacking, [LinearRegression()], 
                      X_train, y_train, X_test, n_folds=1)
        # Wrong verbose value
        assert_raises(ValueError, stacking, [LinearRegression()], 
                      X_train, y_train, X_test, verbose=25)
                      
        # Internal function model_action
        assert_raises(ValueError, model_action, LinearRegression(),
                      X_train, y_train, X_test, sample_weight=None,
                      action='abc', transform=None)

        # X_test is None when mode != 'oof'
        assert_raises(ValueError, stacking, [LinearRegression()],
                      X_train, y_train, None, mode='oof_pred_bag')
                      
    #---------------------------------------------------------------------------
    # Testing parameter warnings
    #---------------------------------------------------------------------------
    def test_warnings(self):
        # Parameters specific for classification are ignored if regression=True
        assert_warns(UserWarning, stacking, [LinearRegression()], 
                      X_train, y_train, X_test, regression=True, 
                      needs_proba=True)
                      
        assert_warns(UserWarning, stacking, [LinearRegression()], 
                      X_train, y_train, X_test, regression=True, 
                      stratified=True)
                      
        assert_warns(UserWarning, stacking, [LinearRegression()], 
                      X_train, y_train, X_test, regression=True, 
                      needs_proba=True, stratified=True)
                      
    #---------------------------------------------------------------------------
    # Test if model has no 'get_params'
    #---------------------------------------------------------------------------
    def test_oof_pred_mode_no_get_params(self):
    
        S_train_1 = np.ones(X_train.shape[0]).reshape(-1, 1)
        S_test_1 = np.ones(X_test.shape[0]).reshape(-1, 1)

        models = [MinimalEstimator()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred', random_state = 0, verbose = 0)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
    
    #--------------------------------------------------------------------------
    # Test inconsistent data shape or type
    #--------------------------------------------------------------------------
    def test_inconsistent_data(self):
        # nan or inf in y
        y_train_nan = y_train.copy()
        y_train_nan[0] = np.nan
        assert_raises(ValueError, stacking, [LinearRegression()], 
                      X_train, y_train_nan, X_test)
                      
        # y has two or more columns
        assert_raises(ValueError, stacking, [LinearRegression()], 
                      X_train, np.c_[y_train, y_train], X_test)
                      
        # X_train and y_train shape nismatch
        assert_raises(ValueError, stacking, [LinearRegression()], 
                      X_train, y_train[:10], X_test)

    #---------------------------------------------------------------------------
    # Test small input
    #---------------------------------------------------------------------------

    def test_small_input(self):
        """
        This is `test_oof_pred_bag_mode` with small input data
        Train: 20 examples
        Test: 10 examples
        """
        S_test_temp = np.zeros((X_test[:10].shape[0], n_folds))
        kf = KFold(n_splits = n_folds, shuffle = False, random_state = None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train[:20], y_train[:20])):
            # Split data and target
            X_tr = X_train[:20][tr_index]
            y_tr = y_train[:20][tr_index]
            X_te = X_train[:20][te_index]
            y_te = y_train[:20][te_index]
            model = LinearRegression()
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test[:10])
        S_test_1 = np.mean(S_test_temp, axis = 1).reshape(-1, 1)
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train[:20], y = y_train[:20], cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train[:20], y_train[:20], X_test[:10], 
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred_bag', random_state = 0, verbose = 0)

        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)

        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)

    #---------------------------------------------------------------------------
    # Mode 'oof', X_test=None
    #---------------------------------------------------------------------------

    def test_oof_mode_with_none(self):

        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds,
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        S_test_1 = None

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, None,
            regression = True, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof', random_state = 0, verbose = 0)

        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)

        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)

    #---------------------------------------------------------------------------
    # All default values (mode='oof_pred_bag')
    #---------------------------------------------------------------------------

    def test_all_defaults(self):

        # Override global n_folds=5, because default value in stacking function is 4
        n_folds=4

        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        kf = KFold(n_splits = n_folds, shuffle = False, random_state = None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = LinearRegression()
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1 = np.mean(S_test_temp, axis = 1).reshape(-1, 1)

        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds,
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, save_dir=temp_dir)

        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
        S_train_3 = S[0]
        S_test_3 = S[1]

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

