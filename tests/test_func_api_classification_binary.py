#-------------------------------------------------------------------------------
# Exactly the same as multiclass but ``n_classes=2``
# and class name is ``TestFuncClassificationBinary``
#-------------------------------------------------------------------------------
# !!! cross_val_predict uses stratified split
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

import os
import glob
import numpy as np
import scipy.stats as st
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from vecstack import stacking

from sklearn.multiclass import OneVsRestClassifier

n_classes = 2
n_folds = 5
temp_dir = 'tmpdw35lg54ms80eb42'

X, y = make_classification(n_samples = 500, n_features = 5, n_informative = 3, n_redundant = 1, 
                           n_classes = n_classes, flip_y = 0, random_state = 0)
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


# Create 4-dim data
np.random.seed(42)
X_train_4d = np.random.normal(size=(400, 8, 8, 3))
X_test_4d = np.random.normal(size=(100, 8, 8, 3))
y_train_4d = np.random.randint(n_classes, size=400)

# Reshape 4-dim to 2-dim
X_train_4d_unrolled = X_train_4d.reshape(X_train_4d.shape[0], -1)
X_test_4d_unrolled = X_test_4d.reshape(X_test_4d.shape[0], -1)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class LogisticRegressionUnrolled(LogisticRegression):
    """
    For tests related to N-dim input.
    Estimator accepts N-dim array and reshape it to 2-dim array
    """
    def fit(self, X, y):
        return super(LogisticRegressionUnrolled, self).fit(X.reshape(X.shape[0], -1), y)

    def predict(self, X):
        return super(LogisticRegressionUnrolled, self).predict(X.reshape(X.shape[0], -1))

    def predict_proba(self, X):
        return super(LogisticRegressionUnrolled, self).predict_proba(X.reshape(X.shape[0], -1))


class OneVsRestClassifierUnrolled(OneVsRestClassifier):
    """
    Just to avoid data shape checks
    """
    def fit(self, X, y):
        return super(OneVsRestClassifierUnrolled, self).fit(X.reshape(X.shape[0], -1), y)

    def predict(self, X):
        return super(OneVsRestClassifierUnrolled, self).predict(X.reshape(X.shape[0], -1))

    def predict_proba(self, X):
        return super(OneVsRestClassifierUnrolled, self).predict_proba(X.reshape(X.shape[0], -1))

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class TestFuncClassificationBinary(unittest.TestCase):

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
    # Test returned and saved arrays in each mode (parameter <mode>)
    # Here we also test parameter <stratified> 
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # Predict labels
    #---------------------------------------------------------------------------

    def test_oof_pred_mode(self):

        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        _ = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir=temp_dir, 
            mode = 'oof_pred', random_state = 0, verbose = 0, stratified = True)
            
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

        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
        S_test_1 = None

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir=temp_dir, 
            mode = 'oof', random_state = 0, verbose = 0, stratified = True)
            
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

        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1 = None
        _ = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir=temp_dir, 
            mode = 'pred', random_state = 0, verbose = 0, stratified = True)
            
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
        # Usind StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1 = st.mode(S_test_temp, axis = 1, keepdims=True)[0]
    
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred_bag', random_state = 0, verbose = 0, stratified = True)
            
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
        # Usind StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1 = st.mode(S_test_temp, axis = 1, keepdims=True)[0]
    
        S_train_1 = None

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'pred_bag', random_state = 0, verbose = 0, stratified = True)
            
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
    # Predict proba
    #---------------------------------------------------------------------------
        
    def test_oof_pred_mode_proba(self):

        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict_proba')
        _ = model.fit(X_train, y_train)
        S_test_1 = model.predict_proba(X_test)

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, stratified = True,
            mode = 'oof_pred', random_state = 0, verbose = 0, needs_proba = True, save_dir=temp_dir)
            
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
        
    def test_oof_mode_proba(self):

        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict_proba')
        S_test_1 = None

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, stratified = True, 
            mode = 'oof', random_state = 0, verbose = 0, needs_proba = True, save_dir=temp_dir)
            
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
        
    def test_pred_mode_proba(self):

        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1 = None
        _ = model.fit(X_train, y_train)
        S_test_1 = model.predict_proba(X_test)

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, stratified = True, 
            mode = 'pred', random_state = 0, verbose = 0, needs_proba = True, save_dir=temp_dir)
            
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
        
    def test_oof_pred_bag_mode_proba(self):
        
        S_test_1 = np.zeros((X_test.shape[0], n_classes))
        S_test_temp = np.zeros((X_test.shape[0], n_folds * n_classes))
        # Using StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
            _ = model.fit(X_tr, y_tr)
            col_slice_fold = slice(fold_counter * n_classes, fold_counter * n_classes + n_classes)
            S_test_temp[:, col_slice_fold] = model.predict_proba(X_test)
        for class_id in range(n_classes):
            S_test_1[:, class_id] = np.mean(S_test_temp[:, class_id::n_classes], axis = 1)
    
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict_proba')

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred_bag', random_state = 0, verbose = 0, stratified = True, needs_proba = True)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
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
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
            _ = model.fit(X_tr, y_tr)
            col_slice_fold = slice(fold_counter * n_classes, fold_counter * n_classes + n_classes)
            S_test_temp[:, col_slice_fold] = model.predict_proba(X_test)
        for class_id in range(n_classes):
            S_test_1[:, class_id] = np.mean(S_test_temp[:, class_id::n_classes], axis = 1)
    
        S_train_1 = None

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'pred_bag', random_state = 0, verbose = 0, stratified = True, needs_proba = True)
            
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
            model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1 = st.mode(S_test_temp, axis = 1, keepdims=True)[0]
    
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        # !!! Important. Here we pass CV-generator not number of folds <cv = kf>
        S_train_1 = cross_val_predict(model, X_train, y = y_train, cv = kf, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = True, save_dir=temp_dir,
            mode = 'oof_pred_bag', random_state = 0, verbose = 0, stratified = True)
            
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
    # Test <metric> parameter and its default values depending on <regression> parameter
    # Labels
    # Important. We use <greater_is_better = True> in <make_scorer> for any error function
    # because we need raw scores (without minus sign)
    #---------------------------------------------------------------------------
    def test_oof_mode_metric(self):

        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        scorer = make_scorer(accuracy_score)
        scores = cross_val_score(model, X_train, y = y_train, cv = n_folds, 
            scoring = scorer, n_jobs = 1, verbose = 0)
        mean_str_1 = '%.8f' % np.mean(scores)
        std_str_1 = '%.8f' % np.std(scores)
        

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))]
        S_train, S_test = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, save_dir=temp_dir, 
            mode = 'oof', random_state = 0, verbose = 0, stratified = True)
            
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
    
    #---------------------------------------------------------------------------
    # Test <metric> parameter and its default values depending on <regression> parameter
    # Proba
    # Important. We use <greater_is_better = True> in <make_scorer> for any error function
    # because we need raw scores (without minus sign)
    #---------------------------------------------------------------------------
    def test_oof_mode_metric_proba(self):

        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        scorer = make_scorer(log_loss, response_method='predict_proba')
        scores = cross_val_score(model, X_train, y = y_train, cv = n_folds, 
            scoring = scorer, n_jobs = 1, verbose = 0)
        mean_str_1 = '%.8f' % np.mean(scores)
        std_str_1 = '%.8f' % np.std(scores)
        

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))]
        S_train, S_test = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, save_dir=temp_dir, 
            mode = 'oof', random_state = 0, verbose = 0, stratified = True, 
            needs_proba = True)
            
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

        # Model a
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
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

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')),
                  GaussianNB()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir=temp_dir, 
            mode = 'oof_pred', random_state = 0, verbose = 0, stratified = True)
            
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
        # Usind StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1_a = st.mode(S_test_temp, axis = 1, keepdims=True)[0]
    
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1_a = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
            
        # Model b
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        # Usind StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = GaussianNB()
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1_b = st.mode(S_test_temp, axis = 1, keepdims=True)[0]
    
        model = GaussianNB()
        S_train_1_b = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)
            
        S_train_1 = np.c_[S_train_1_a, S_train_1_b]
        S_test_1 = np.c_[S_test_1_a, S_test_1_b]

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')),
                  GaussianNB()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred_bag', random_state = 0, verbose = 0, stratified = True)
            
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
        
        
    def test_oof_pred_mode_proba_2_models(self):

        # Model a
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
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

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')),
                  GaussianNB()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, stratified = True,
            mode = 'oof_pred', random_state = 0, verbose = 0, needs_proba = True, save_dir=temp_dir)
            
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
        
        
    def test_oof_pred_bag_mode_proba_2_models(self):
        
        # Model a
        S_test_1_a = np.zeros((X_test.shape[0], n_classes))
        S_test_temp = np.zeros((X_test.shape[0], n_folds * n_classes))
        # Using StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
            _ = model.fit(X_tr, y_tr)
            col_slice_fold = slice(fold_counter * n_classes, fold_counter * n_classes + n_classes)
            S_test_temp[:, col_slice_fold] = model.predict_proba(X_test)
        for class_id in range(n_classes):
            S_test_1_a[:, class_id] = np.mean(S_test_temp[:, class_id::n_classes], axis = 1)
    
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1_a = cross_val_predict(model, X_train, y = y_train, cv = n_folds, 
            n_jobs = 1, verbose = 0, method = 'predict_proba')
            
        # Model b
        S_test_1_b = np.zeros((X_test.shape[0], n_classes))
        S_test_temp = np.zeros((X_test.shape[0], n_folds * n_classes))
        # Using StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = None)
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
        
        

        models = [OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')),
                  GaussianNB()]
        S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test, 
            regression = False, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred_bag', random_state = 0, verbose = 0, stratified = True, needs_proba = True)
            
        # Load OOF from file
        # Normally if cleaning is performed there is only one .npy file at given moment
        # But if we have no cleaning there may be more then one file so we take the latest
        file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1] # take the latest file
        S = np.load(file_name, allow_pickle=True)
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

    def test_N_dim_input(self):
        """
        This is `test_oof_pred_bag_mode` function with `LogisticRegressionUnrolled` estimator
        """
        S_test_temp = np.zeros((X_test_4d_unrolled.shape[0], n_folds))
        # Usind StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train_4d_unrolled, y_train_4d)):
            # Split data and target
            X_tr = X_train_4d_unrolled[tr_index]
            y_tr = y_train_4d[tr_index]
            X_te = X_train_4d_unrolled[te_index]
            y_te = y_train_4d[te_index]
            model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
            _ = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test_4d_unrolled)
        S_test_1 = st.mode(S_test_temp, axis = 1, keepdims=True)[0]
    
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1 = cross_val_predict(model, X_train_4d_unrolled, y = y_train_4d, cv = n_folds,
            n_jobs = 1, verbose = 0, method = 'predict').reshape(-1, 1)

        models = [OneVsRestClassifierUnrolled(LogisticRegressionUnrolled(random_state=0, solver='liblinear'))]
        S_train_2, S_test_2 = stacking(models, X_train_4d, y_train_4d, X_test_4d,
            regression = False, n_folds = n_folds, shuffle = False, save_dir=temp_dir,
            mode = 'oof_pred_bag', random_state = 0, verbose = 0, stratified = True)

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

