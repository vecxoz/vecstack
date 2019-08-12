# -----------------------------------------------------------------------------
# Main concept for testing returned arrays:
# 1). create ground truth e.g. with cross_val_predict
# 2). run vecstack
# 3). compare returned arrays with ground truth
# -----------------------------------------------------------------------------

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
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.estimator_checks import check_estimator
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from vecstack import StackingTransformer

n_folds = 5
temp_dir = 'tmpdw35lg54ms80eb42'

boston = load_boston()
X, y = boston.data, boston.target
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


# -----------------------------------------------------------------------------
# Scikit-learn INcompatible estimator 
# -----------------------------------------------------------------------------

class IncompatibleEstimator:
    """ This estimator is not compatible with scikit-learn API. For tests only.
    - has no ``get_params`` and ``set_params`` methods
    - can not be cloned when ``safe=True``: ``clone(IncompatibleEstimator(), safe=True)`` raises TypeError
    - can not be used in Pipeline
    """
    def __init__(self, random_state=0):
        self.random_state = random_state
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.ones(X.shape[0])
    def predict_proba(self, X):
        return np.zeros(X.shape[0])
        
# -----------------------------------------------------------------------------
# Scikit-learn compatible estimator wich does not support ``sample_weight``
# -----------------------------------------------------------------------------

class EstimatorWithoutSampleWeight(BaseEstimator, RegressorMixin):
    """ This estimator does not support ``sample_weight``. For tests only.
    """
    def __init__(self, random_state=0):
        self.random_state = random_state
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.ones(X.shape[0])
    def predict_proba(self, X):
        return np.zeros(X.shape[0])

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class TestSklearnRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            os.mkdir(temp_dir)
        except:
            print('Unable to create temp dir')
            
    @classmethod
    def tearDownClass(cls):
        try:
            files = glob.glob(os.path.join(temp_dir, '*.*'))
            for file in files:
                os.remove(file)
            os.rmdir(temp_dir)
        except:
            print('Unable to remove temp file or temp dir')

    # -------------------------------------------------------------------------
    # Main scikit-learn compatibility test
    # -------------------------------------------------------------------------
    
    def test_sklearn_compatibility(self):
        check_estimator(StackingTransformer)
    
    # -------------------------------------------------------------------------
    # Test returned arrays in variant B
    # -------------------------------------------------------------------------

    def test_variant_B(self):
        # reference
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        S_train_2 = stack.transform(X_train)
        S_test_2 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(X_train, y_train)
        S_test_3 = stack.transform(X_test)
        
        # compare
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
    
    # -------------------------------------------------------------------------
    # Test returned arrays in variant A
    # -------------------------------------------------------------------------
    
    def test_variant_A(self):
        
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        kf = KFold(n_splits=n_folds, shuffle=False, random_state=0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            # X_te = X_train[te_index]
            # y_te = y_train[te_index]
            model = LinearRegression()
            model = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1 = np.mean(S_test_temp, axis=1).reshape(-1, 1)
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)

        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='A', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        S_train_2 = stack.transform(X_train)
        S_test_2 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(X_train, y_train)
        S_test_3 = stack.transform(X_test)

        # compare
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)

    # -------------------------------------------------------------------------
    # Testing ``sample_weight`` all ones
    # -------------------------------------------------------------------------
    def test_variant_B_sample_weight_one(self):
    
        sw = np.ones(len(y_train))
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict', 
                                      fit_params={'sample_weight': sw}).reshape(-1, 1)
        model = model.fit(X_train, y_train, sample_weight=sw)
        S_test_1 = model.predict(X_test).reshape(-1, 1)
        
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train, sample_weight=sw)
        S_train_2 = stack.transform(X_train)
        S_test_2 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(X_train, y_train, sample_weight=sw)
        S_test_3 = stack.transform(X_test)

        # compare
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
    
    # -------------------------------------------------------------------------
    # Test ``sample_weight`` all random
    # -------------------------------------------------------------------------
    def test_variant_B_sample_weight_random(self):
    
        np.random.seed(0)
        sw = np.random.rand(len(y_train))
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict', 
                                      fit_params={'sample_weight': sw}).reshape(-1, 1)
        model = model.fit(X_train, y_train, sample_weight=sw)
        S_test_1 = model.predict(X_test).reshape(-1, 1)
        
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train, sample_weight=sw)
        S_train_2 = stack.transform(X_train)
        S_test_2 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(X_train, y_train, sample_weight=sw)
        S_test_3 = stack.transform(X_test)

        # compare
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)

    #---------------------------------------------------------------------------
    # Testing ``transform_target`` and ``transform_pred`` parameters
    #---------------------------------------------------------------------------
    
    def test_variant_B_transformations(self):
    
        model = LinearRegression()
        S_train_1 = np.expm1(cross_val_predict(model, X_train,
                                               y=np.log1p(y_train), 
                                               cv=n_folds, n_jobs=1,
                                               verbose=0,
                                               method='predict')).reshape(-1, 1)
        model = model.fit(X_train, np.log1p(y_train))
        S_test_1 = np.expm1(model.predict(X_test)).reshape(-1, 1)
        
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    transform_target=np.log1p,
                                    transform_pred=np.expm1,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        S_train_2 = stack.transform(X_train)
        S_test_2 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(X_train, y_train)
        S_test_3 = stack.transform(X_test)

        # compare
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)

    # -------------------------------------------------------------------------
    # Test ``verbose`` parameter. Variant B
    # -------------------------------------------------------------------------
    def test_variant_B_verbose(self):
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)
        
        # verbose=0
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        S_train_2 = stack.transform(X_train)
        S_test_2 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(X_train, y_train)
        S_test_3 = stack.transform(X_test)
        
        # verbose=1
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=1)
        stack = stack.fit(X_train, y_train)
        S_train_4 = stack.transform(X_train)
        S_test_4 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_5 = stack.fit_transform(X_train, y_train)
        S_test_5 = stack.transform(X_test)
        
        # verbose=2
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=2)
        stack = stack.fit(X_train, y_train)
        S_train_6 = stack.transform(X_train)
        S_test_6 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_7 = stack.fit_transform(X_train, y_train)
        S_test_7 = stack.transform(X_test)
        

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
        
    # -------------------------------------------------------------------------
    # Test ``verbose`` parameter. Variant A
    # -------------------------------------------------------------------------
    def test_variant_A_verbose(self):
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        kf = KFold(n_splits=n_folds, shuffle=False, random_state=0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            # X_te = X_train[te_index]
            # y_te = y_train[te_index]
            model = LinearRegression()
            model = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1 = np.mean(S_test_temp, axis=1).reshape(-1, 1)
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
                                      
        # verbose=0
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='A', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        S_train_2 = stack.transform(X_train)
        S_test_2 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(X_train, y_train)
        S_test_3 = stack.transform(X_test)
        
        # verbose=1
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='A', random_state=0,
                                    verbose=1)
        stack = stack.fit(X_train, y_train)
        S_train_4 = stack.transform(X_train)
        S_test_4 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_5 = stack.fit_transform(X_train, y_train)
        S_test_5 = stack.transform(X_test)
        
        # verbose=2
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='A', random_state=0,
                                    verbose=2)
        stack = stack.fit(X_train, y_train)
        S_train_6 = stack.transform(X_train)
        S_test_6 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_7 = stack.fit_transform(X_train, y_train)
        S_test_7 = stack.transform(X_test)
        

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
    
    # -------------------------------------------------------------------------
    # Test default metric and scores. 1 estimator
    # ``metric`` parameter and its default values depends on ``regression`` parameter.
    # Important. We use ``greater_is_better=True`` in ``make_scorer``
    # for any error function because we need raw scores (without minus sign)
    # -------------------------------------------------------------------------
    def test_default_metric_and_scores_1_estimator(self):

        model = LinearRegression()
        scorer = make_scorer(mean_absolute_error)
        scores_1 = cross_val_score(model, X_train, y=y_train,
                                   cv=n_folds, scoring=scorer,
                                   n_jobs=1, verbose=0)
        
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    metric=None,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        scores_2 = stack.scores_[0].copy()
        
        # mean and std
        mean_1 = np.mean(scores_1)
        std_1 = np.std(scores_1)
        
        mean_2 = stack.mean_std_[0][1]
        std_2 = stack.mean_std_[0][2]
        
        assert_array_equal(scores_1, scores_2)
        assert_equal(mean_1, mean_2)
        assert_equal(std_1, std_2)
            
        # fit_transform
        # also check refitting already fitted transformer
        _ = stack.fit_transform(X_train, y_train)
        scores_3 = stack.scores_[0].copy()
        
        mean_3 = stack.mean_std_[0][1]
        std_3 = stack.mean_std_[0][2]
        
        assert_array_equal(scores_1, scores_3)
        assert_equal(mean_1, mean_3)
        assert_equal(std_1, std_3)
        
    # -------------------------------------------------------------------------
    # Test custom metric and scores. 1 estimator
    # -------------------------------------------------------------------------
    def test_custom_metric_and_scores_1_estimator(self):

        model = LinearRegression()
        scorer = make_scorer(mean_squared_error)
        scores_1 = cross_val_score(model, X_train, y=y_train,
                                   cv=n_folds, scoring=scorer,
                                   n_jobs=1, verbose=0)
        
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    metric=mean_squared_error,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        scores_2 = stack.scores_[0].copy()
        
        # mean and std
        mean_1 = np.mean(scores_1)
        std_1 = np.std(scores_1)
        
        mean_2 = stack.mean_std_[0][1]
        std_2 = stack.mean_std_[0][2]
        
        assert_array_equal(scores_1, scores_2)
        assert_equal(mean_1, mean_2)
        assert_equal(std_1, std_2)
            
        # fit_transform
        # also check refitting already fitted transformer
        _ = stack.fit_transform(X_train, y_train)
        scores_3 = stack.scores_[0].copy()
        
        mean_3 = stack.mean_std_[0][1]
        std_3 = stack.mean_std_[0][2]
        
        assert_array_equal(scores_1, scores_3)
        assert_equal(mean_1, mean_3)
        assert_equal(std_1, std_3)
        
    # -------------------------------------------------------------------------
    # Test default metric and scores. 2 estimator
    # -------------------------------------------------------------------------
    def test_default_metric_and_scores_2_estimators(self):

        scorer = make_scorer(mean_absolute_error)
        model = LinearRegression()
        scores_1_e1 = cross_val_score(model, X_train, y=y_train,
                                      cv=n_folds, scoring=scorer,
                                      n_jobs=1, verbose=0)
        model = Ridge(random_state=0)
        scores_1_e2 = cross_val_score(model, X_train, y=y_train,
                                      cv=n_folds, scoring=scorer,
                                      n_jobs=1, verbose=0)
        
        # fit then transform
        estimators = [('lr', LinearRegression()),
                      ('ridge', Ridge(random_state=0))]
        stack = StackingTransformer(estimators, regression=True,
                                    metric=None,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        # create copy to avoid array replacement after refit
        scores_2_e1 = stack.scores_[0].copy()
        scores_2_e2 = stack.scores_[1].copy()
        
        # mean and std
        mean_1_e1 = np.mean(scores_1_e1)
        std_1_e1 = np.std(scores_1_e1)
        mean_1_e2 = np.mean(scores_1_e2)
        std_1_e2 = np.std(scores_1_e2)
        
        mean_2_e1 = stack.mean_std_[0][1]
        std_2_e1 = stack.mean_std_[0][2]
        mean_2_e2 = stack.mean_std_[1][1]
        std_2_e2 = stack.mean_std_[1][2]
        
        assert_array_equal(scores_1_e1, scores_2_e1)
        assert_array_equal(scores_1_e2, scores_2_e2)
        assert_equal(mean_1_e1, mean_2_e1)
        assert_equal(mean_1_e2, mean_2_e2)
        assert_equal(std_1_e1, std_2_e1)
        assert_equal(std_1_e2, std_2_e2)
        
        # fit_transform
        # also check refitting already fitted transformer
        _ = stack.fit_transform(X_train, y_train)
        scores_3_e1 = stack.scores_[0].copy()
        scores_3_e2 = stack.scores_[1].copy()
        
        mean_3_e1 = stack.mean_std_[0][1]
        std_3_e1 = stack.mean_std_[0][2]
        mean_3_e2 = stack.mean_std_[1][1]
        std_3_e2 = stack.mean_std_[1][2]
        
        assert_array_equal(scores_1_e1, scores_3_e1)
        assert_array_equal(scores_1_e2, scores_3_e2)
        assert_equal(mean_1_e1, mean_3_e1)
        assert_equal(mean_1_e2, mean_3_e2)
        assert_equal(std_1_e1, std_3_e1)
        assert_equal(std_1_e2, std_3_e2)
        
    # -------------------------------------------------------------------------
    # Test custom metric and scores. 2 estimator
    # -------------------------------------------------------------------------
    def test_custom_metric_and_scores_2_estimators(self):

        scorer = make_scorer(mean_squared_error)
        model = LinearRegression()
        scores_1_e1 = cross_val_score(model, X_train, y=y_train,
                                      cv=n_folds, scoring=scorer,
                                      n_jobs=1, verbose=0)
        model = Ridge(random_state=0)
        scores_1_e2 = cross_val_score(model, X_train, y=y_train,
                                      cv=n_folds, scoring=scorer,
                                      n_jobs=1, verbose=0)
        
        # fit then transform
        estimators = [('lr', LinearRegression()),
                      ('ridge', Ridge(random_state=0))]
        stack = StackingTransformer(estimators, regression=True,
                                    metric=mean_squared_error,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        # create copy to avoid array replacement after refit
        scores_2_e1 = stack.scores_[0].copy()
        scores_2_e2 = stack.scores_[1].copy()
        
        # mean and std
        mean_1_e1 = np.mean(scores_1_e1)
        std_1_e1 = np.std(scores_1_e1)
        mean_1_e2 = np.mean(scores_1_e2)
        std_1_e2 = np.std(scores_1_e2)
        
        mean_2_e1 = stack.mean_std_[0][1]
        std_2_e1 = stack.mean_std_[0][2]
        mean_2_e2 = stack.mean_std_[1][1]
        std_2_e2 = stack.mean_std_[1][2]
        
        assert_array_equal(scores_1_e1, scores_2_e1)
        assert_array_equal(scores_1_e2, scores_2_e2)
        assert_equal(mean_1_e1, mean_2_e1)
        assert_equal(mean_1_e2, mean_2_e2)
        assert_equal(std_1_e1, std_2_e1)
        assert_equal(std_1_e2, std_2_e2)
        
        # fit_transform
        # also check refitting already fitted transformer
        _ = stack.fit_transform(X_train, y_train)
        scores_3_e1 = stack.scores_[0].copy()
        scores_3_e2 = stack.scores_[1].copy()
        
        mean_3_e1 = stack.mean_std_[0][1]
        std_3_e1 = stack.mean_std_[0][2]
        mean_3_e2 = stack.mean_std_[1][1]
        std_3_e2 = stack.mean_std_[1][2]
        
        assert_array_equal(scores_1_e1, scores_3_e1)
        assert_array_equal(scores_1_e2, scores_3_e2)
        assert_equal(mean_1_e1, mean_3_e1)
        assert_equal(mean_1_e2, mean_3_e2)
        assert_equal(std_1_e1, std_3_e1)
        assert_equal(std_1_e2, std_3_e2)
    
    # -------------------------------------------------------------------------
    # Test several estimators in one run. Variant B
    # -------------------------------------------------------------------------
    def test_variant_B_2_estimators(self):
    
        model = LinearRegression()
        S_train_1_a = cross_val_predict(model, X_train, y=y_train,
                                        cv=n_folds, n_jobs=1, verbose=0,
                                        method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1_a = model.predict(X_test).reshape(-1, 1)
        
        model = Ridge(random_state=0)
        S_train_1_b = cross_val_predict(model, X_train, y=y_train,
                                        cv=n_folds, n_jobs=1, verbose=0,
                                        method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1_b = model.predict(X_test).reshape(-1, 1)
        
        S_train_1 = np.c_[S_train_1_a, S_train_1_b]
        S_test_1 = np.c_[S_test_1_a, S_test_1_b]
        
        # fit then transform
        estimators = [('lr', LinearRegression()),
                      ('ridge', Ridge(random_state=0))]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        S_train_2 = stack.transform(X_train)
        S_test_2 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(X_train, y_train)
        S_test_3 = stack.transform(X_test)

        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    # -------------------------------------------------------------------------
    # Test several estimators in one run. Variant A
    # -------------------------------------------------------------------------

    def test_variant_A_2_estimators(self):
        
        # Model a
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        kf = KFold(n_splits=n_folds, shuffle=False, random_state=0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            # X_te = X_train[te_index]
            # y_te = y_train[te_index]
            model = LinearRegression()
            model = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1_a = np.mean(S_test_temp, axis=1).reshape(-1, 1)
    
        model = LinearRegression()
        S_train_1_a = cross_val_predict(model, X_train, y=y_train,
                                        cv=n_folds, n_jobs=1, verbose=0,
                                        method='predict').reshape(-1, 1)
            
        # Model b
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        kf = KFold(n_splits=n_folds, shuffle=False, random_state=0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            # X_te = X_train[te_index]
            # y_te = y_train[te_index]
            model = Ridge(random_state=0)
            model = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1_b = np.mean(S_test_temp, axis=1).reshape(-1, 1)
    
        model = Ridge(random_state=0)
        S_train_1_b = cross_val_predict(model, X_train, y=y_train,
                                        cv=n_folds, n_jobs=1, verbose=0,
                                        method='predict').reshape(-1, 1)
            
        S_train_1 = np.c_[S_train_1_a, S_train_1_b]
        S_test_1 = np.c_[S_test_1_a, S_test_1_b]
        
        # fit then transform
        estimators = [('lr', LinearRegression()),
                      ('ridge', Ridge(random_state=0))]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='A', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        S_train_2 = stack.transform(X_train)
        S_test_2 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(X_train, y_train)
        S_test_3 = stack.transform(X_test)
        
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    
    # -------------------------------------------------------------------------
    # Testing sparse types CSR, CSC, COO
    # -------------------------------------------------------------------------

    def test_variant_B_sparse_csr(self):
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, csr_matrix(X_train), y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
        model = model.fit(csr_matrix(X_train), y_train)
        S_test_1 = model.predict(csr_matrix(X_test)).reshape(-1, 1)
        
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(csr_matrix(X_train), y_train)
        S_train_2 = stack.transform(csr_matrix(X_train))
        S_test_2 = stack.transform(csr_matrix(X_test))
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(csr_matrix(X_train), y_train)
        S_test_3 = stack.transform(csr_matrix(X_test))
        
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    def test_variant_B_sparse_csc(self):
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, csc_matrix(X_train), y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
        model = model.fit(csc_matrix(X_train), y_train)
        S_test_1 = model.predict(csc_matrix(X_test)).reshape(-1, 1)
        
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(csc_matrix(X_train), y_train)
        S_train_2 = stack.transform(csc_matrix(X_train))
        S_test_2 = stack.transform(csc_matrix(X_test))
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(csc_matrix(X_train), y_train)
        S_test_3 = stack.transform(csc_matrix(X_test))
        
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    def test_variant_B_sparse_coo(self):
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, coo_matrix(X_train), y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
        model = model.fit(coo_matrix(X_train), y_train)
        S_test_1 = model.predict(coo_matrix(X_test)).reshape(-1, 1)
        
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(coo_matrix(X_train), y_train)
        S_train_2 = stack.transform(coo_matrix(X_train))
        S_test_2 = stack.transform(coo_matrix(X_test))
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(coo_matrix(X_train), y_train)
        S_test_3 = stack.transform(coo_matrix(X_test))
        
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)

    #---------------------------------------------------------------------------
    # Testing X_train -> SCR, X_test -> COO
    #---------------------------------------------------------------------------
    def test_variant_B_sparse_csr_coo(self):
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, csr_matrix(X_train), y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
        model = model.fit(csr_matrix(X_train), y_train)
        S_test_1 = model.predict(coo_matrix(X_test)).reshape(-1, 1)
        
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(csr_matrix(X_train), y_train)
        S_train_2 = stack.transform(csr_matrix(X_train))
        S_test_2 = stack.transform(coo_matrix(X_test))
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(csr_matrix(X_train), y_train)
        S_test_3 = stack.transform(coo_matrix(X_test))
        
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
    
    # -------------------------------------------------------------------------
    # Testing X_train -> SCR, X_test -> Dense
    # -------------------------------------------------------------------------
    def test_variant_B_sparse_csr_dense(self):
    
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, csr_matrix(X_train), y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
        model = model.fit(csr_matrix(X_train), y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)
        
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(csr_matrix(X_train), y_train)
        S_train_2 = stack.transform(csr_matrix(X_train))
        S_test_2 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(csr_matrix(X_train), y_train)
        S_test_3 = stack.transform(X_test)
        
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)

    # -------------------------------------------------------------------------
    # Test default (dummy) regressor
    # -------------------------------------------------------------------------
    def test_variant_B_default_regressor(self):
        # reference
        model = DummyRegressor(strategy='constant', constant=5.5)
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        # fit then transform
        stack = StackingTransformer(estimators=None, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        S_train_2 = stack.transform(X_train)
        S_test_2 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(X_train, y_train)
        S_test_3 = stack.transform(X_test)
        
        # compare
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    #-------------------------------------------------------------------------------
    # Test inconsistent data shape or type
    #-------------------------------------------------------------------------------
    def test_inconsistent_data(self):
    
        stack = StackingTransformer()
        
        # ----
        
        y_train_nan = y_train.copy()
        y_train_nan[0] = np.nan
        
        # nan or inf in y
        assert_raises(ValueError, stack.fit, X_train, y_train_nan)

        # y has two or more columns
        assert_raises(ValueError, stack.fit, X_train, np.c_[y_train, y_train])
                
        # X and y 1st dimention mismatch
        assert_raises(ValueError, stack.fit, X_train, y_train[:10])
        
        # ----
        
        sample_weight = np.random.rand(len(y_train))
        sample_weight_nan = sample_weight.copy()
        sample_weight_nan[0] = np.nan
        
        # nan or inf in sample_weight
        assert_raises(ValueError, stack.fit, X_train, y_train, sample_weight_nan)
        
        # sample_weight has two or more columns
        assert_raises(ValueError, stack.fit, X_train, y_train, np.c_[sample_weight, sample_weight])
        
        # X and sample_weight 1st dimention mismatch
        assert_raises(ValueError, stack.fit, X_train, y_train, sample_weight[:10])
        
    # -------------------------------------------------------------------------
    # Inconsistent estimator names
    # -------------------------------------------------------------------------
    def test_inconsistent_estimator_names(self):
    
        # Names are not unique
        estimators = [('lr', LinearRegression()),
                      ('lr', Ridge(random_state=0))]
    
        stack = StackingTransformer(estimators)
        assert_raises(ValueError, stack.fit, X_train, y_train)
        
        # Names conflict with constructor arguments
        estimators = [('lr', LinearRegression()),
                      ('n_folds', Ridge(random_state=0))]
    
        stack = StackingTransformer(estimators)
        assert_raises(ValueError, stack.fit, X_train, y_train)
        
        # Names contain double underscore
        estimators = [('lr', LinearRegression()),
                      ('ridge__ridge', Ridge(random_state=0))]
    
        stack = StackingTransformer(estimators)
        assert_raises(ValueError, stack.fit, X_train, y_train)

    # -------------------------------------------------------------------------
    # Testing parameter exceptions
    # -------------------------------------------------------------------------
    def test_parameter_exceptions(self):
        
        # Wrong variant
        stack = StackingTransformer(variant='C')
        assert_raises(ValueError, stack.fit, X_train, y_train)
                
        # n_folds is not int
        stack = StackingTransformer(n_folds='4')
        assert_raises(ValueError, stack.fit, X_train, y_train)

        # n_folds is less than 2
        stack = StackingTransformer(n_folds=1)
        assert_raises(ValueError, stack.fit, X_train, y_train)
        
        # Wrong verbose value
        stack = StackingTransformer(verbose=25)
        assert_raises(ValueError, stack.fit, X_train, y_train)
 
        # Internal function _estimator_action
        stack = StackingTransformer()
        assert_raises(ValueError, stack._estimator_action, 
                      LinearRegression(), X_train, y_train, X_test,
                      sample_weight=None, action='abc', transform=None)
                      
        # Estimator list is empty
        stack = StackingTransformer(estimators=[])
        assert_raises(ValueError, stack.fit, X_train, y_train)
        
        # Estimator does not support ``sample_weight``
        estimators = [('no_sample_weight', EstimatorWithoutSampleWeight())]
        stack = StackingTransformer(estimators)
        smaple_weight = np.random.rand(len(y_train))
        # If we do NOT pass sample_weight to StackingTransfrmer.fit it do NOT raise
        # (normally works with this estimator)
        assert_raises(AssertionError, assert_raises, ValueError, stack.fit, X_train, y_train)
        # If we pass sample_weight to StackingTransfrmer.fit it raises
        assert_raises(ValueError, stack.fit, X_train, y_train, smaple_weight)

    # -------------------------------------------------------------------------
    # Testing parameter warnings
    # -------------------------------------------------------------------------
    def test_parameter_warnings(self):
        # Parameters specific for classification are ignored if regression=True
        stack = StackingTransformer(regression=True, needs_proba=True)
        assert_warns(UserWarning, stack.fit, X_train, y_train)
        
        stack = StackingTransformer(regression=True, stratified=True)
        assert_warns(UserWarning, stack.fit, X_train, y_train)
        
        stack = StackingTransformer(regression=True, needs_proba=True, stratified=True)
        assert_warns(UserWarning, stack.fit, X_train, y_train)
        
    # -------------------------------------------------------------------------
    # Test incompatible estimator
    # -------------------------------------------------------------------------
    def test_incompatible_estimator(self):
        estimators = [('ie', IncompatibleEstimator())]
        stack = StackingTransformer(estimators)
        assert_raises(TypeError, stack.fit, X_train, y_train)
        
    # -------------------------------------------------------------------------
    # Test ability of fitted StackingTransformer instance to be pickled
    # -------------------------------------------------------------------------
    def test_pickle(self):
        # reference
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        # fit
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        
        # save
        _ = joblib.dump(stack, os.path.join(temp_dir, 'stack.pkl'))
        
        # load
        stack_loaded = joblib.load(os.path.join(temp_dir, 'stack.pkl'))
        
        # transform using loaded instance
        S_train_2 = stack_loaded.transform(X_train)
        S_test_2 = stack_loaded.transform(X_test)
            
        # refit loaded instance with fit_transform
        S_train_3 = stack_loaded.fit_transform(X_train, y_train)
        S_test_3 = stack_loaded.transform(X_test)
        
        # compare
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    # -------------------------------------------------------------------------
    # Test ``StackingTransformer.is_train_set`` public method
    # -------------------------------------------------------------------------
    def test_is_train_set(self):
        # fit
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        
        assert_equal(stack.is_train_set(X_train), True)
        assert_equal(stack.is_train_set(X_test), False)
    
    # -------------------------------------------------------------------------
    # Test fallback parameter ``is_train_set`` of ``transform`` method
    # -------------------------------------------------------------------------
    def test_fallback_parameter_of_transform_method(self):
        # reference
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        # We explicitly tell that this is train set and it actually IS train set
        # MUST work as usual
        S_train_2 = stack.transform(X_train, is_train_set=True)
        S_test_2 = stack.transform(X_test)
        
        # compare
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        # We explicitly tell that this is train set but it actually is NOT train set (different shape)
        # MUST raise
        assert_raises(ValueError, stack.transform, X_train[:10], is_train_set=True)
    
    # -------------------------------------------------------------------------
    # Test Pipeline and ability to set parameters of nested estimators
    # -------------------------------------------------------------------------
    def test_pipeline(self):
        # reference
        model = LinearRegression(fit_intercept=True)
        S_train_1_lr = cross_val_predict(model, X_train, y=y_train,
                                         cv=n_folds, n_jobs=1, verbose=0,
                                         method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1_lr = model.predict(X_test).reshape(-1, 1)
        
        model = DecisionTreeRegressor(random_state=0, max_depth=2)
        S_train_1_ridge = cross_val_predict(model, X_train, y=y_train,
                                            cv=n_folds, n_jobs=1, verbose=0,
                                            method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1_ridge = model.predict(X_test).reshape(-1, 1)
        
        S_train_1 = np.c_[S_train_1_lr, S_train_1_ridge]
        S_test_1 = np.c_[S_test_1_lr, S_test_1_ridge]
        
        model = Ridge(random_state=0, alpha=2)
        model = model.fit(S_train_1, y_train)
        y_pred_1 = model.predict(S_test_1)

        # We intentionally set different parameter values to reset them
        # later using ``set_params`` method
        # We have 4 parameters which differs from reference:
        # ``fit_intercept``, ``max_depth``, ``variant``, and ``alpha``
        estimators = [('lr', LinearRegression(fit_intercept=False)),
                      ('tree', DecisionTreeRegressor(random_state=0, max_depth=4))]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='A', random_state=0,
                                    verbose=0)
        ridge = Ridge(random_state=0, alpha=7)
        
        steps = [('stack', stack),
                 ('ridge', ridge)]
                 
        pipe = Pipeline(steps)
        
        pipe = pipe.fit(X_train, y_train)
        y_pred_2 = pipe.predict(X_test)
        
        # Here we expect that final predictions are different 
        # because we've set different parameters
        assert_raises(AssertionError, assert_array_equal, y_pred_1, y_pred_2)
        
        # Reset original parameters used in reference
        pipe = pipe.set_params(stack__lr__fit_intercept=True,
                               stack__tree__max_depth=2,
                               stack__variant='B',
                               ridge__alpha=2)
                               
        pipe = pipe.fit(X_train, y_train)
        y_pred_3 = pipe.predict(X_test)
        
        # Here we expect that final predictions are equal
        assert_array_equal(y_pred_1, y_pred_3)
        
    # -------------------------------------------------------------------------
    # Test FeatureUnion and ability to set parameters of nested estimators
    # -------------------------------------------------------------------------
    def test_feature_union(self):
        # reference
        model = LinearRegression(fit_intercept=True)
        S_train_1_lr = cross_val_predict(model, X_train, y=y_train,
                                         cv=n_folds, n_jobs=1, verbose=0,
                                         method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1_lr = model.predict(X_test).reshape(-1, 1)
        
        model = Ridge(random_state=0, alpha=2)
        S_train_1_ridge = cross_val_predict(model, X_train, y=y_train,
                                            cv=n_folds, n_jobs=1, verbose=0,
                                            method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1_ridge = model.predict(X_test).reshape(-1, 1)
        
        kbest = SelectKBest(score_func=f_regression, k=5)
        kbest = kbest.fit(X_train, y_train)
        S_train_1_kbest = kbest.transform(X_train)
        S_test_1_kbest = kbest.transform(X_test)
        
        S_train_1 = np.c_[S_train_1_lr, S_train_1_ridge, S_train_1_kbest]
        S_test_1 = np.c_[S_test_1_lr, S_test_1_ridge, S_test_1_kbest]
        
        
        # We intentionally set different parameter values to reset them
        # later using ``set_params`` method
        # We have 4 parameters which differs from reference:
        # ``fit_intercept``, ``alpha``, ``variant``, and ``k``
        estimators = [('lr', LinearRegression(fit_intercept=False)),
                      ('ridge', Ridge(random_state=0, alpha=7))]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='A', random_state=0,
                                    verbose=0)
        kbest = SelectKBest(score_func=f_regression, k=9)
        
        transformers = [('stack', stack),
                        ('kbest', kbest)]
                 
        union = FeatureUnion(transformers)
        
        union = union.fit(X_train, y_train)
        S_train_2 = union.transform(X_train)
        S_test_2 = union.transform(X_test)
        
        # Here we expect that final predictions are different
        # because we've set different parameters
        assert_raises(AssertionError, assert_array_equal, S_train_1, S_train_2)
        assert_raises(AssertionError, assert_array_equal, S_test_1, S_test_2)
        
        # Reset original parameters used in reference
        union = union.set_params(stack__lr__fit_intercept=True,
                                 stack__ridge__alpha=2,
                                 stack__variant='B',
                                 kbest__k=5)
                               
        union = union.fit(X_train, y_train)
        S_train_3 = union.transform(X_train)
        S_test_3 = union.transform(X_test)
        
        # Here we expect that final predictions are equal
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
        
    # -------------------------------------------------------------------------
    # Test Pipeline with FeatureUnion
    # -------------------------------------------------------------------------
    def test_pipeline_with_feature_union(self):
        # reference
        model = LinearRegression(fit_intercept=True)
        S_train_1_lr = cross_val_predict(model, X_train, y=y_train,
                                         cv=n_folds, n_jobs=1, verbose=0,
                                         method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1_lr = model.predict(X_test).reshape(-1, 1)
        
        model = Ridge(random_state=0, alpha=2)
        S_train_1_ridge = cross_val_predict(model, X_train, y=y_train,
                                            cv=n_folds, n_jobs=1, verbose=0,
                                            method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1_ridge = model.predict(X_test).reshape(-1, 1)
        
        kbest = SelectKBest(score_func=f_regression, k=5)
        kbest = kbest.fit(X_train, y_train)
        S_train_1_kbest = kbest.transform(X_train)
        S_test_1_kbest = kbest.transform(X_test)
        
        S_train_1 = np.c_[S_train_1_lr, S_train_1_ridge, S_train_1_kbest]
        S_test_1 = np.c_[S_test_1_lr, S_test_1_ridge, S_test_1_kbest]
        
        model = DecisionTreeRegressor(random_state=0, max_depth=2) 
        model = model.fit(S_train_1, y_train)
        y_pred_1 = model.predict(S_test_1)

        # We intentionally set different parameter values to reset them
        # later using ``set_params`` method
        # We have 5 parameters which differs from reference:
        # ``fit_intercept``, ``alpha``, ``variant``, ``k``, and ``max_depth``
        estimators = [('lr', LinearRegression(fit_intercept=False)),
                      ('ridge', Ridge(random_state=0, alpha=7))]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='A', random_state=0,
                                    verbose=0)
                                    
        kbest = SelectKBest(score_func=f_regression, k=9)
        
        transformers = [('stack', stack),
                        ('kbest', kbest)]
                 
        union = FeatureUnion(transformers)
        
        tree = DecisionTreeRegressor(random_state=0, max_depth=4)
        
        steps = [('union', union),
                 ('tree', tree)]
                 
        pipe = Pipeline(steps)
        
        pipe = pipe.fit(X_train, y_train)
        y_pred_2 = pipe.predict(X_test)
        
        # Here we expect that final predictions are different 
        # because we've set different parameters
        assert_raises(AssertionError, assert_array_equal, y_pred_1, y_pred_2)
        
        # Reset original parameters used in reference
        pipe = pipe.set_params(union__stack__lr__fit_intercept=True,
                               union__stack__ridge__alpha=2,
                               union__stack__variant='B',
                               union__kbest__k=5,
                               tree__max_depth=2)
                               
        pipe = pipe.fit(X_train, y_train)
        y_pred_3 = pipe.predict(X_test)
        
        # Here we expect that final predictions are equal
        assert_array_equal(y_pred_1, y_pred_3)
        
    # -------------------------------------------------------------------------
    # Test GridSearchCV and RandomizedGridSearchCV with Pipeline and FeatureUnion
    # -------------------------------------------------------------------------
    def test_grid_search_with_pipeline_and_feature_union(self):
        # reference
        model = LinearRegression(fit_intercept=True)
        S_train_1_lr = cross_val_predict(model, X_train, y=y_train,
                                         cv=n_folds, n_jobs=1, verbose=0,
                                         method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1_lr = model.predict(X_test).reshape(-1, 1)
        
        model = Ridge(random_state=0, alpha=2)
        S_train_1_ridge = cross_val_predict(model, X_train, y=y_train,
                                            cv=n_folds, n_jobs=1, verbose=0,
                                            method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1_ridge = model.predict(X_test).reshape(-1, 1)
        
        kbest = SelectKBest(score_func=f_regression, k=5)
        kbest = kbest.fit(X_train, y_train)
        S_train_1_kbest = kbest.transform(X_train)
        S_test_1_kbest = kbest.transform(X_test)
        
        S_train_1 = np.c_[S_train_1_lr, S_train_1_ridge, S_train_1_kbest]
        S_test_1 = np.c_[S_test_1_lr, S_test_1_ridge, S_test_1_kbest]
        
        model = DecisionTreeRegressor(random_state=0, max_depth=2)
        model = model.fit(S_train_1, y_train)
        y_pred_1 = model.predict(S_test_1)

        # We intentionally set different parameter values to search best in grid
        # We have 5 parameters which differs from reference:
        # ``fit_intercept``, ``alpha``, ``variant``, ``k``, and ``max_depth``
        estimators = [('lr', LinearRegression(fit_intercept=False)),
                      ('ridge', Ridge(random_state=0, alpha=7))]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='A', random_state=0,
                                    verbose=0)
                                    
        kbest = SelectKBest(score_func=f_regression, k=9)
        
        transformers = [('stack', stack),
                        ('kbest', kbest)]
                 
        union = FeatureUnion(transformers)
        
        tree = DecisionTreeRegressor(random_state=0, max_depth=4)
        
        steps = [('union', union),
                 ('tree', tree)]
                 
        pipe = Pipeline(steps)
        
        # Parameter search space
        params = {'union__stack__lr__fit_intercept': [True],
                  'union__stack__ridge__alpha': [2],
                  'union__stack__variant' : ['B'],
                  'union__kbest__k': [1, 5],
                  'tree__max_depth': [2]}

        # Exhaustive search
        grid = GridSearchCV(pipe, params, n_jobs=1)
        grid = grid.fit(X_train, y_train)
        y_pred_2 = grid.predict(X_test)
        
        assert_array_equal(y_pred_1, y_pred_2)
        
        # Random search
        # We set such ``n_iter`` that it gives exhaustive search to get the same result as reference
        rand_grid = RandomizedSearchCV(pipe, params, random_state=0, n_iter=2)
        rand_grid = rand_grid.fit(X_train, y_train)
        y_pred_3 = rand_grid.predict(X_test)
        
        assert_array_equal(y_pred_1, y_pred_3)
        
    # -------------------------------------------------------------------------
    # Test ``shuffle`` and ``random_state``. Variant B
    # -------------------------------------------------------------------------
    def test_variant_B_shuffle_and_random_state(self):
        # reference
        model = LinearRegression()
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        # !!! below we pass KFold object (not a number of folds)
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=kf, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=True,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        S_train_2 = stack.transform(X_train)
        S_test_2 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(X_train, y_train)
        S_test_3 = stack.transform(X_test)
        
        # compare
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
    
    # -------------------------------------------------------------------------
    # Test ``shuffle`` and ``random_state``. Variant A
    # -------------------------------------------------------------------------
    def test_variant_A_shuffle_and_random_state(self):
        
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            # X_te = X_train[te_index]
            # y_te = y_train[te_index]
            model = LinearRegression()
            model = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1 = np.mean(S_test_temp, axis=1).reshape(-1, 1)
    
        model = LinearRegression()
        # !!! below we pass KFold object (not a number of folds)
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=kf, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)

        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=True,
                                    variant='A', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        S_train_2 = stack.transform(X_train)
        S_test_2 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(X_train, y_train)
        S_test_3 = stack.transform(X_test)

        # compare
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)
       
    # -------------------------------------------------------------------------
    # Test ``_get_footprint`` method
    # -------------------------------------------------------------------------
    def test_get_footprint(self):
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        # ``X`` argument is correct - must NOT raise
        assert_raises(AssertionError, assert_raises, ValueError, stack._get_footprint, X_train)
        # ``X`` argument is INcorrect - MUST raise
        assert_raises(ValueError, stack._get_footprint, 5)
        
    # -------------------------------------------------------------------------
    # Test ``_check_identity`` method
    # -------------------------------------------------------------------------
    def test_check_identity(self):
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        # ``X`` argument is correct - must NOT raise
        assert_raises(AssertionError, assert_raises, ValueError, stack._check_identity, X_train)
        # ``X`` argument is INcorrect - MUST raise
        assert_raises(ValueError, stack._check_identity, 5)

    # -------------------------------------------------------------------------
    # Test ``_random_choice`` method
    # -------------------------------------------------------------------------
    def test_random_choice(self):
        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        # very large range - must NOT raise
        assert_raises(AssertionError, assert_raises, ValueError, stack._random_choice, 19999999999, 1000)
        # ``size`` is less than ``n`` - must NOT raise
        assert_raises(AssertionError, assert_raises, ValueError, stack._random_choice, 200, 20)
        # ``size`` is greater than ``n`` - MUST raise
        assert_raises(ValueError, stack._random_choice, 20, 200)
        
    # -------------------------------------------------------------------------
    # Test case where X_test has the same shape as X_train
    # but contains different data
    # -------------------------------------------------------------------------
    def test_x_test_has_same_shape(self):
        # reference
        X_test_same_shape = np.r_[X_test[:101], X_test[:101], X_test[:101], X_test[:101]]
        
        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test_same_shape).reshape(-1, 1)

        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        S_train_2 = stack.transform(X_train)
        S_test_2 = stack.transform(X_test_same_shape)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(X_train, y_train)
        S_test_3 = stack.transform(X_test_same_shape)
        
        # compare
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)
        
        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)

    # -------------------------------------------------------------------------
    # Test small input
    # -------------------------------------------------------------------------

    def test_small_input(self):
        """
        This is `test_variant_A` with small input data
        Train: 20 examples
        Test: 10 examples
        """
        S_test_temp = np.zeros((X_test[:10].shape[0], n_folds))
        kf = KFold(n_splits=n_folds, shuffle=False, random_state=0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train[:20], y_train[:20])):
            # Split data and target
            X_tr = X_train[:20][tr_index]
            y_tr = y_train[:20][tr_index]
            # X_te = X_train[:20][te_index]
            # y_te = y_train[:20][te_index]
            model = LinearRegression()
            model = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test[:10])
        S_test_1 = np.mean(S_test_temp, axis=1).reshape(-1, 1)

        model = LinearRegression()
        S_train_1 = cross_val_predict(model, X_train[:20], y=y_train[:20],
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)

        # fit then transform
        estimators = [('lr', LinearRegression())]
        stack = StackingTransformer(estimators, regression=True,
                                    n_folds=n_folds, shuffle=False,
                                    variant='A', random_state=0,
                                    verbose=0)
        stack = stack.fit(X_train[:20], y_train[:20])
        S_train_2 = stack.transform(X_train[:20])
        S_test_2 = stack.transform(X_test[:10])

        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(X_train[:20], y_train[:20])
        S_test_3 = stack.transform(X_test[:10])

        # compare
        assert_array_equal(S_train_1, S_train_2)
        assert_array_equal(S_test_1, S_test_2)

        assert_array_equal(S_train_1, S_train_3)
        assert_array_equal(S_test_1, S_test_3)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

