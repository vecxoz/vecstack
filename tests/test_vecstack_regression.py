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
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from vecstack import stacking

n_folds = 5

boston = load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class TestRegression(unittest.TestCase):

    def tearDown(self):
        # Remove files after each test
        files = glob.glob('*.npy')
        files.extend(glob.glob('*.txt'))
        for file in files:
            os.remove(file)
            
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
            regression = True, n_folds = n_folds, shuffle = False, save_dir = '.',
            mode = 'oof_pred', random_state = 0, verbose = 0)
            
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

        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        S_test_1 = None

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir = '.', 
            mode = 'oof', random_state = 0, verbose = 0)
            
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
    
        model = LinearRegression()
        S_train_1 = None
        _ = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir = '.', 
            mode = 'pred', random_state = 0, verbose = 0)
            
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
        kf = KFold(n_splits = n_folds, shuffle = False, random_state = 0)
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
            regression = True, n_folds = n_folds, shuffle = False, save_dir = '.',
            mode = 'oof_pred_bag', random_state = 0, verbose = 0)
            
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
        kf = KFold(n_splits = n_folds, shuffle = False, random_state = 0)
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
            regression = True, n_folds = n_folds, shuffle = False, save_dir = '.',
            mode = 'pred_bag', random_state = 0, verbose = 0)
            
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
    # Testing <sample_weight> all ones
    #---------------------------------------------------------------------------
    def test_oof_pred_mode_sample_weight_one(self):
    
        sw = np.ones(len(y_train))
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict', 
            fit_params = {'sample_weight': sw}).reshape(-1, 1)
        _ = model.fit(X_train, y_train, sample_weight = sw)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir = '.',
            mode = 'oof_pred', random_state = 0, verbose = 0,
            sample_weight = sw)
            
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
    # Test <sample_weight> all random
    #---------------------------------------------------------------------------
    def test_oof_pred_mode_sample_weight_random(self):
    
        np.random.seed(0)
        sw = np.random.rand(len(y_train))
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict', 
            fit_params = {'sample_weight': sw}).reshape(-1, 1)
        _ = model.fit(X_train, y_train, sample_weight = sw)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        models = [LinearRegression()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir = '.',
            mode = 'oof_pred', random_state = 0, verbose = 0,
            sample_weight = sw)
            
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
            regression = True, n_folds = n_folds, shuffle = False, save_dir = '.',
            mode = 'oof_pred', random_state = 0, verbose = 0,
            transform_target = np.log1p, transform_pred = np.expm1)
            
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
            regression = True, n_folds = n_folds, shuffle = False, save_dir = '.', 
            mode = 'oof_pred', random_state = 0, verbose = 0)

        models = [LinearRegression()]
        S_train_3, S_test_3 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir = '.', 
            mode = 'oof_pred', random_state = 0, verbose = 1)
            
        models = [LinearRegression()]
        S_train_4, S_test_4 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir = '.', 
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
            regression = True, n_folds = n_folds, save_dir = '.', 
            mode = 'oof', random_state = 0, verbose = 0)
            
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
    
        model = LinearRegression()
        S_train_1_a = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(X_train, y_train)
        S_test_1_a = model.predict(X_test).reshape(-1, 1)
        
        model = SGDRegressor(random_state = 0)
        S_train_1_b = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(X_train, y_train)
        S_test_1_b = model.predict(X_test).reshape(-1, 1)
        
        S_train_1 = np.c_[S_train_1_a, S_train_1_b]
        S_test_1 = np.c_[S_test_1_a, S_test_1_b]

        models = [LinearRegression(),
                  SGDRegressor(random_state = 0)]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir = '.',
            mode = 'oof_pred', random_state = 0, verbose = 0)
            
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
        kf = KFold(n_splits = n_folds, shuffle = False, random_state = 0)
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
        kf = KFold(n_splits = n_folds, shuffle = False, random_state = 0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = SGDRegressor(random_state = 0)
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1_b = np.mean(S_test_temp, axis = 1).reshape(-1, 1)
    
        model = SGDRegressor(random_state = 0)
        S_train_1_b = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
            
        S_train_1 = np.c_[S_train_1_a, S_train_1_b]
        S_test_1 = np.c_[S_test_1_a, S_test_1_b]
        

        models = [LinearRegression(),
                  SGDRegressor(random_state = 0)]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = True, n_folds = n_folds, shuffle = False, save_dir = '.',
            mode = 'oof_pred_bag', random_state = 0, verbose = 0)
            
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
        
        
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

