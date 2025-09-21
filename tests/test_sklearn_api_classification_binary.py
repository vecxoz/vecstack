# -----------------------------------------------------------------------------
# Exactly the same as multiclass but ``n_classes=2``
# and class name is ``TestSklearnClassificationBinary``
# -----------------------------------------------------------------------------
# !!! cross_val_predict uses stratified split
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

# import os
# import glob
import numpy as np
import scipy.stats as st
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from vecstack import StackingTransformer

from sklearn.multiclass import OneVsRestClassifier

n_classes = 2
n_folds = 5
# temp_dir = 'tmpdw35lg54ms80eb42'

X, y = make_classification(n_samples=500, n_features=5,
                           n_informative=3, n_redundant=1,
                           n_classes=n_classes, flip_y=0,
                           random_state=0)

# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size=0.2,
#                                                     random_state=0)


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

def roc_auc_score_universal(y_true, y_pred):
    """ROC AUC metric for both binary and multiclass classification.
    
    Parameters
    ----------
    y_true - 1d numpy array
        True class labels
    y_pred - 2d numpy array
        Predicted probabilities for each class
    """
    ohe = OneHotEncoder(sparse_output=False)
    y_true = ohe.fit_transform(y_true.reshape(-1, 1))
    #@@@@
    if len(y_pred.shape) == 1:
        y_pred = np.c_[y_pred, y_pred]
        y_pred[:, 0] = 1 - y_pred[:, 1]
    #@@@@
    auc_score = roc_auc_score(y_true, y_pred)
    return auc_score

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class TestSklearnClassificationBinary(unittest.TestCase):

    # -------------------------------------------------------------------------
    # Test returned arrays in variant B. Labels
    # -------------------------------------------------------------------------

    def test_variant_B_labels(self):
        # reference
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        # fit then transform
        estimators = [('logit', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')))]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    stratified=True, verbose=0)
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
    # Test returned arrays in variant A. Labels
    # -------------------------------------------------------------------------
    
    def test_variant_A_labels(self):
        
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        # Using StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            # X_te = X_train[te_index]
            # y_te = y_train[te_index]
            model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
            model = model.fit(X_tr, y_tr)
            S_test_temp[:, fold_counter] = model.predict(X_test)
        S_test_1 = st.mode(S_test_temp, axis=1, keepdims=True)[0]
    
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)

        # fit then transform
        estimators = [('logit', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')))]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='A', random_state=0,
                                    stratified=True, verbose=0)
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
    # Test returned arrays in variant B. Probabilities
    # -------------------------------------------------------------------------

    def test_variant_B_proba(self):
        # reference
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict_proba')
        model = model.fit(X_train, y_train)
        S_test_1 = model.predict_proba(X_test)

        # fit then transform
        estimators = [('logit', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')))]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    stratified=True, needs_proba=True,
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
    # Test returned arrays in variant A. Probabilities
    # -------------------------------------------------------------------------
    
    def test_variant_A_proba(self):
        
        S_test_1 = np.zeros((X_test.shape[0], n_classes))
        S_test_temp = np.zeros((X_test.shape[0], n_folds * n_classes))
        # Using StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            # X_te = X_train[te_index]
            # y_te = y_train[te_index]
            model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
            model = model.fit(X_tr, y_tr)
            col_slice_fold = slice(fold_counter * n_classes, fold_counter * n_classes + n_classes)
            S_test_temp[:, col_slice_fold] = model.predict_proba(X_test)
        for class_id in range(n_classes):
            S_test_1[:, class_id] = np.mean(S_test_temp[:, class_id::n_classes], axis=1)
    
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict_proba')

        # fit then transform
        estimators = [('logit', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')))]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='A', random_state=0,
                                    stratified=True, needs_proba=True,
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
        
    #---------------------------------------------------------------------------
    # Test ``shuffle`` and ``random_state`` parameters in variant A
    #---------------------------------------------------------------------------
    def test_variant_A_proba_shuffle_random_state(self):
        
        S_test_1 = np.zeros((X_test.shape[0], n_classes))
        S_test_temp = np.zeros((X_test.shape[0], n_folds * n_classes))
        # Using StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            # X_te = X_train[te_index]
            # y_te = y_train[te_index]
            model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
            model = model.fit(X_tr, y_tr)
            col_slice_fold = slice(fold_counter * n_classes, fold_counter * n_classes + n_classes)
            S_test_temp[:, col_slice_fold] = model.predict_proba(X_test)
        for class_id in range(n_classes):
            S_test_1[:, class_id] = np.mean(S_test_temp[:, class_id::n_classes], axis=1)
    
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        # !!! Important. Here we pass CV-generator ``cv=kf`` not number of folds
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=kf, n_jobs=1, verbose=0,
                                      method='predict_proba')

        # fit then transform
        estimators = [('logit', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')))]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=True,
                                    variant='A', random_state=0,
                                    stratified=True, needs_proba=True,
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
    # Test default metric and scores. Labels
    # ``metric`` parameter and its default values depends on ``regression`` parameter.
    # Important. We use ``greater_is_better=True`` in ``make_scorer``
    # for any error function because we need raw scores (without minus sign)
    # -------------------------------------------------------------------------
    def test_default_metric_and_scores_labels(self):

        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        scorer = make_scorer(accuracy_score)
        scores_1 = cross_val_score(model, X_train, y=y_train,
                                   cv=n_folds, scoring=scorer,
                                   n_jobs=1, verbose=0)
        
        # fit then transform
        estimators = [('logit', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')))]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    stratified=True, verbose=0)
        stack = stack.fit(X_train, y_train)
        scores_2 = stack.scores_[0].copy()
            
        # fit_transform
        # also check refitting already fitted transformer
        _ = stack.fit_transform(X_train, y_train)
        scores_3 = stack.scores_[0].copy()
        
        assert_array_equal(scores_1, scores_2)
        assert_array_equal(scores_1, scores_3)
        
        # mean and std
        mean_1 = np.mean(scores_1)
        std_1 = np.std(scores_1)
        
        mean_2 = stack.mean_std_[0][1]
        std_2 = stack.mean_std_[0][2]
        
        assert_equal(mean_1, mean_2)
        assert_equal(std_1, std_2)
        
    # -------------------------------------------------------------------------
    # Test custom metric and scores. Labels
    # -------------------------------------------------------------------------
    def test_custom_metric_and_scores_labels(self):

        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        scorer = make_scorer(zero_one_loss)
        scores_1 = cross_val_score(model, X_train, y=y_train,
                                   cv=n_folds, scoring=scorer,
                                   n_jobs=1, verbose=0)
        
        # fit then transform
        estimators = [('logit', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')))]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    stratified=True, metric=zero_one_loss,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        scores_2 = stack.scores_[0].copy()
            
        # fit_transform
        # also check refitting already fitted transformer
        _ = stack.fit_transform(X_train, y_train)
        scores_3 = stack.scores_[0].copy()
        
        assert_array_equal(scores_1, scores_2)
        assert_array_equal(scores_1, scores_3)
        
        # mean and std
        mean_1 = np.mean(scores_1)
        std_1 = np.std(scores_1)
        
        mean_2 = stack.mean_std_[0][1]
        std_2 = stack.mean_std_[0][2]
        
        assert_equal(mean_1, mean_2)
        assert_equal(std_1, std_2)
        
    # -------------------------------------------------------------------------
    # Test default metric and scores. Probabilities
    # -------------------------------------------------------------------------
    def test_default_metric_and_scores_proba(self):

        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        scorer = make_scorer(log_loss, response_method='predict_proba')
        scores_1 = cross_val_score(model, X_train, y=y_train,
                                   cv=n_folds, scoring=scorer,
                                   n_jobs=1, verbose=0)
        
        # fit then transform
        estimators = [('logit', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')))]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    stratified=True, needs_proba=True,
                                    verbose=0)
        stack = stack.fit(X_train, y_train)
        scores_2 = stack.scores_[0].copy()
            
        # fit_transform
        # also check refitting already fitted transformer
        _ = stack.fit_transform(X_train, y_train)
        scores_3 = stack.scores_[0].copy()
        
        assert_array_equal(scores_1, scores_2)
        assert_array_equal(scores_1, scores_3)
        
        # mean and std
        mean_1 = np.mean(scores_1)
        std_1 = np.std(scores_1)
        
        mean_2 = stack.mean_std_[0][1]
        std_2 = stack.mean_std_[0][2]
        
        assert_equal(mean_1, mean_2)
        assert_equal(std_1, std_2)
        
    # -------------------------------------------------------------------------
    # Test custom metric and scores. Probabilities
    # -------------------------------------------------------------------------
    def test_custom_metric_and_scores_proba(self):

        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        scorer = make_scorer(roc_auc_score_universal, response_method='predict_proba')
        scores_1 = cross_val_score(model, X_train, y=y_train,
                                   cv=n_folds, scoring=scorer,
                                   n_jobs=1, verbose=0)
        
        # fit then transform
        estimators = [('logit', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')))]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    stratified=True, needs_proba=True,
                                    metric=roc_auc_score_universal, verbose=0)
        stack = stack.fit(X_train, y_train)
        scores_2 = stack.scores_[0].copy()
            
        # fit_transform
        # also check refitting already fitted transformer
        _ = stack.fit_transform(X_train, y_train)
        scores_3 = stack.scores_[0].copy()
        
        assert_array_equal(scores_1, scores_2)
        assert_array_equal(scores_1, scores_3)
        
        # mean and std
        mean_1 = np.mean(scores_1)
        std_1 = np.std(scores_1)
        
        mean_2 = stack.mean_std_[0][1]
        std_2 = stack.mean_std_[0][2]
        
        assert_equal(mean_1, mean_2)
        assert_equal(std_1, std_2)
        
    # -------------------------------------------------------------------------
    # Test several estimators in one run. Variant B. Labels
    # -------------------------------------------------------------------------

    def test_variant_B_2_estimators_labels(self):
    
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1_e1 = cross_val_predict(model, X_train, y=y_train,
                                        cv=n_folds, n_jobs=1, verbose=0,
                                        method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1_e1 = model.predict(X_test).reshape(-1, 1)
        
        model = GaussianNB()
        S_train_1_e2 = cross_val_predict(model, X_train, y=y_train,
                                        cv=n_folds, n_jobs=1, verbose=0,
                                        method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1_e2 = model.predict(X_test).reshape(-1, 1)
        
        S_train_1 = np.c_[S_train_1_e1, S_train_1_e2]
        S_test_1 = np.c_[S_test_1_e1, S_test_1_e2]
        
        # fit then transform
        estimators = [('logit', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))),
                      ('bayes', GaussianNB())]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    stratified=True, verbose=0)
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
    # Test several estimators in one run. Variant B. Probabilities
    # -------------------------------------------------------------------------

    def test_variant_B_2_estimators_proba(self):
    
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1_e1 = cross_val_predict(model, X_train, y=y_train,
                                        cv=n_folds, n_jobs=1, verbose=0,
                                        method='predict_proba')
        model = model.fit(X_train, y_train)
        S_test_1_e1 = model.predict_proba(X_test)
        
        model = GaussianNB()
        S_train_1_e2 = cross_val_predict(model, X_train, y=y_train,
                                        cv=n_folds, n_jobs=1, verbose=0,
                                        method='predict_proba')
        model = model.fit(X_train, y_train)
        S_test_1_e2 = model.predict_proba(X_test)
        
        S_train_1 = np.c_[S_train_1_e1, S_train_1_e2]
        S_test_1 = np.c_[S_test_1_e1, S_test_1_e2]
        
        # fit then transform
        estimators = [('logit', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))),
                      ('bayes', GaussianNB())]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    stratified=True, needs_proba=True,
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
    # Test several estimators in one run. Variant A. Labels
    # -------------------------------------------------------------------------
    
    def test_variant_A_2_estimators_labels(self):
        
        # Estimator 1
        S_test_temp_e1 = np.zeros((X_test.shape[0], n_folds))
        # Using StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            # X_te = X_train[te_index]
            # y_te = y_train[te_index]
            model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
            model = model.fit(X_tr, y_tr)
            S_test_temp_e1[:, fold_counter] = model.predict(X_test)
        S_test_1_e1 = st.mode(S_test_temp_e1, axis=1, keepdims=True)[0]
    
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1_e1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
                                      
        # Estimator 2
        S_test_temp_e2 = np.zeros((X_test.shape[0], n_folds))
        # Using StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            # X_te = X_train[te_index]
            # y_te = y_train[te_index]
            model = GaussianNB()
            model = model.fit(X_tr, y_tr)
            S_test_temp_e2[:, fold_counter] = model.predict(X_test)
        S_test_1_e2 = st.mode(S_test_temp_e2, axis=1, keepdims=True)[0]
    
        model = GaussianNB()
        S_train_1_e2 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)

        S_train_1 = np.c_[S_train_1_e1, S_train_1_e2]
        S_test_1 = np.c_[S_test_1_e1, S_test_1_e2]
        
        # fit then transform
        estimators = [('logit', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))),
                      ('bayes', GaussianNB())]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='A', random_state=0,
                                    stratified=True, verbose=0)
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
    # Test several estimators in one run. Variant A. Probabilities
    # -------------------------------------------------------------------------
    
    def test_variant_A_2_estimators_proba(self):
        
        # Estimator 1
        S_test_1_e1 = np.zeros((X_test.shape[0], n_classes))
        S_test_temp_e1 = np.zeros((X_test.shape[0], n_folds * n_classes))
        # Using StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            # X_te = X_train[te_index]
            # y_te = y_train[te_index]
            model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
            model = model.fit(X_tr, y_tr)
            col_slice_fold = slice(fold_counter * n_classes, fold_counter * n_classes + n_classes)
            S_test_temp_e1[:, col_slice_fold] = model.predict_proba(X_test)
        for class_id in range(n_classes):
            S_test_1_e1[:, class_id] = np.mean(S_test_temp_e1[:, class_id::n_classes], axis=1)
    
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1_e1 = cross_val_predict(model, X_train, y=y_train,
                                         cv=n_folds, n_jobs=1, verbose=0,
                                         method='predict_proba')
                                      
        # Estimator 2
        S_test_1_e2 = np.zeros((X_test.shape[0], n_classes))
        S_test_temp_e2 = np.zeros((X_test.shape[0], n_folds * n_classes))
        # Using StratifiedKFold because by defauld cross_val_predict uses StratifiedKFold
        kf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=None)
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            # X_te = X_train[te_index]
            # y_te = y_train[te_index]
            model = GaussianNB()
            model = model.fit(X_tr, y_tr)
            col_slice_fold = slice(fold_counter * n_classes, fold_counter * n_classes + n_classes)
            S_test_temp_e2[:, col_slice_fold] = model.predict_proba(X_test)
        for class_id in range(n_classes):
            S_test_1_e2[:, class_id] = np.mean(S_test_temp_e2[:, class_id::n_classes], axis=1)
    
        model = GaussianNB()
        S_train_1_e2 = cross_val_predict(model, X_train, y=y_train,
                                         cv=n_folds, n_jobs=1, verbose=0,
                                         method='predict_proba')
                                      
        S_train_1 = np.c_[S_train_1_e1, S_train_1_e2]
        S_test_1 = np.c_[S_test_1_e1, S_test_1_e2]                              

        # fit then transform
        estimators = [('logit', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))),
                      ('bayes', GaussianNB())]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='A', random_state=0,
                                    stratified=True, needs_proba=True,
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
    # Test default (dummy) classifier. Labels
    # -------------------------------------------------------------------------
    def test_variant_B_default_classifier_labels(self):
        # reference
        model = DummyClassifier(strategy='constant', constant=1)
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)

        # fit then transform
        stack = StackingTransformer(estimators=None, regression=False,
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
    # Test default (dummy) classifier. Probabilities
    # -------------------------------------------------------------------------
    def test_variant_B_default_classifier_proba(self):
        # reference
        model = DummyClassifier(strategy='constant', constant=1)
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict_proba')
        model = model.fit(X_train, y_train)
        S_test_1 = model.predict_proba(X_test)

        # fit then transform
        stack = StackingTransformer(estimators=None, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    needs_proba=True, verbose=0)
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
        
    #---------------------------------------------------------------------------
    # Testing ``verbose`` parameter
    #---------------------------------------------------------------------------
    def test_variant_B_verbose(self):
    
        model = OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear'))
        S_train_1 = cross_val_predict(model, X_train, y=y_train,
                                      cv=n_folds, n_jobs=1, verbose=0,
                                      method='predict').reshape(-1, 1)
        model = model.fit(X_train, y_train)
        S_test_1 = model.predict(X_test).reshape(-1, 1)
        
        # verbose=0
        # fit then transform
        estimators = [('lr', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')))]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    stratified=True, verbose=0)
        stack = stack.fit(X_train, y_train)
        S_train_2 = stack.transform(X_train)
        S_test_2 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_3 = stack.fit_transform(X_train, y_train)
        S_test_3 = stack.transform(X_test)
        
        # verbose=1
        # fit then transform
        estimators = [('lr', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')))]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    stratified=True, verbose=1)
        stack = stack.fit(X_train, y_train)
        S_train_4 = stack.transform(X_train)
        S_test_4 = stack.transform(X_test)
            
        # fit_transform
        # also check refitting already fitted transformer
        S_train_5 = stack.fit_transform(X_train, y_train)
        S_test_5 = stack.transform(X_test)
        
        # verbose=2
        # fit then transform
        estimators = [('lr', OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')))]
        stack = StackingTransformer(estimators, regression=False,
                                    n_folds=n_folds, shuffle=False,
                                    variant='B', random_state=0,
                                    stratified=True, verbose=2)
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
        
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

