"""Scikit-learn compatible API for stacking.

Find out how to use:
>>> from vecstack import StackingTransformer
>>> help(StackingTransformer)

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

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

from __future__ import print_function
from __future__ import division

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

import warnings
import numpy as np
import scipy.stats as st
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import has_fit_parameter
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class StackingTransformer(BaseEstimator, TransformerMixin):
    """StackingTransformer. Scikit-learn compatible API for stacking.

    Parameters
    ----------
    estimators : list of tuples, default None
        Base level estimators.
        If None then by default:
            DummyRegressor (predicts constant 5.5) - for regression task
            DummyClassifier (predicts constant 1) - for classification task
        You can use any sklearn-like estimators.
        Each tuple in the list contains arbitrary
            unique name and estimator object, e.g.:
        estimators = [('lr', LinearRegression()),
                      ('ridge', Ridge(random_state=0))]

        Note. According to sklearn convention for binary classification
            task with probabilities estimator must return probabilities
            for each class (i.e. two columns).

    regression : boolean, default True
        If True - perform stacking for regression task,
        if False - perform stacking for classification task

    transform_target : callable, default None
        Function to transform target variable.
        If None - transformation is not used.
        For example, for regression task (if target variable is skewed)
            you can use transformation like ``numpy.log1p``.
            Set ``transform_target=numpy.log1p``
        Usually you want to use respective backward transformation
            for prediction like ``numpy.expm1``.
            Set ``transform_pred=numpy.expm1``
        Caution! Some transformations may give inapplicable results.
            For example, if target variable contains zeros, numpy.log
            gives you -inf. In such case you can use appropriate
            transformation like ``numpy.log1p`` and respective
            backward transformation like ``numpy.expm1``

    transform_pred : callable, default None
        Function to transform prediction.
        If None - transformation is not used.
        If you use transformation for target variable (``transform_target``)
            like ``numpy.log1p``, then using ``transform_pred`` you can specify
            respective backward transformation like ``numpy.expm1``.
        Look at description of parameter ``transform_target``

    variant: str, default 'A'
        Possible values: 'A', 'B'.
        This parameter influences test set predictions only.
        Variant 'A' - predict test set in each fold and find mean (mode)
        Variant 'B' - fit on full train set and predict test set once
        See tutorial for details:
        https://github.com/vecxoz/vecstack/blob/master/ \
            examples/00_stacking_concept_pictures_code.ipynb

    needs_proba: boolean, default False, meaningful only for classification
        Whether to predict probabilities (instead of class labels)
        in classification task.
        Ignored if ``regression=True``

    metric : callable, default None
        Evaluation metric (score function) which is used to calculate
        cross-validation scores.
        If None, then by default:
            sklearn.metrics.mean_absolute_error - for regression
            sklearn.metrics.accuracy_score - for classification
                with class labels
            sklearn.metrics.log_loss - for classification with probabilities
        You can use any appropriate sklearn metric or
            define your own metric like shown below:

        def your_metric(y_true, y_pred):
            # calculate
            return result

    n_folds : int, default 4
        Number of folds in cross-validation

    stratified : boolean, default False, meaningful only for classification
        If True - use stratified folds in cross-validation
        Ignored if ``regression=True``

    shuffle : boolean, default False
        Whether to perform a shuffle before cross-validation split

    random_state : int, default 0
        Random seed used to initiate fold split.
        Same seed and correspondingly same split is used for all estimators.

    verbose : int, default 0
        Level of verbosity.
        0 - show no messages
        1 - for each estimator show mean score
        2 - for each estimator show score for each fold and mean score


    Attributes
    ----------
    estimators_ : list
        List of base estimators (not fitted) passed by user (or default)
    n_estimators_ : int
        Number of base estimators passed by user (or default)
    n_classes_ : int
        Number of classes in classification task.
        ``None`` in regression task.
    models_A_ : list of lists
        List containing n_estimators lists. Each of which contains
        n_folds models (fitted estimators).
        In variant A this models are used to transform (predict)
            both train set and test set
        In variant B this models are used to transform (predict)
            train set only
    models_B_ : list or None
        List containing n_estimators models (fitted estimators).
        In variant A this attribute is None
        In variant B this models are used to transform (predict) test set
    metric_ : callable
        Metric passed by user (or default) which was used
        to compute cross-validation scores
    kf_ : KFold or StratifiedKFold object
        Initialized cross-validation object which was used to split train set
        during fitting
    scores_ : 2d numpy array of shape [n_estimators, n_folds]
        Scores for each fold for each estimator.
    mean_std_ : list of tuples
        Each tuple contains name, mean and std for each estimator.
    train_shape_ : tuple of ints
        Shape of training data
    n_train_examples_ : int
        Number of training examples
    n_features_: int
        Number of features
    train_footprint_ : list of tuples
        Train set footprint which is used to identify train set
        during transform (predict) phase.
        Each tuple containes 3 values: row index, column index, data value

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import mean_absolute_error
    >>> from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    >>> from xgboost import XGBRegressor
    >>> from vecstack import StackingTransformer
    >>>
    >>> # Load demo data
    >>> boston = load_boston()
    >>> X, y = boston.data, boston.target
    >>>
    >>> # Make train/test split
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=0)
    >>>
    >>> # Init 1st level estimators
    >>> estimators_L1 = [('et', ExtraTreesRegressor(random_state=0,
                                                    n_jobs=-1,
                                                    n_estimators=100,
                                                    max_depth=3)),
                         ('rf', RandomForestRegressor(random_state=0,
                                                      n_jobs=-1,
                                                      n_estimators=100,
                                                      max_depth=3)),
                         ('xgb', XGBRegressor(random_state=0,
                                              n_jobs=-1,
                                              learning_rate=0.1,
                                              n_estimators=100,
                                              max_depth=3))]
    >>> # Stacking
    >>> stack = StackingTransformer(estimators=estimators_L1,
                                    regression=True,
                                    shuffle=True,
                                    random_state=0,
                                    verbose=2)
    >>> stack = stack.fit(X_train, y_train)
    >>> S_train = stack.transform(X_train)
    >>> S_test = stack.transform(X_test)
    >>>
    >>> # Use 2nd level estimator to get final prediction
    >>> estimator_L2 = XGBRegressor(random_state=0,
                                    n_jobs=-1,
                                    learning_rate=0.1,
                                    n_estimators=100,
                                    max_depth=3)
    >>> estimator_L2 = estimator_L2.fit(S_train, y_train)
    >>> y_pred = estimator_L2.predict(S_test)
    >>>
    >>> # Final prediction score
    >>> print('Final score: [%.8f]' % mean_absolute_error(y_test, y_pred))
    """
    def __init__(self,
                 estimators=None,
                 regression=True,
                 transform_target=None,
                 transform_pred=None,
                 variant='A',
                 needs_proba=False,
                 metric=None,
                 n_folds=4,
                 stratified=False,
                 shuffle=False,
                 random_state=0,
                 verbose=0):

        self.estimators = estimators
        self.regression = regression
        self.transform_target = transform_target
        self.transform_pred = transform_pred
        self.variant = variant
        self.needs_proba = needs_proba
        self.metric = metric
        self.n_folds = n_folds
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def fit(self, X, y, sample_weight=None):
        """Fit all base estimators.

        Parameters
        ----------
        X : 2d numpy array or sparse matrix of shape [n_samples, n_features]
            Training data
        y : 1d numpy array of shape [n_samples]
            Target values.
        sample_weight : 1d numpy array of shape [n_samples]
            Individual weights for each sample.
            Passed to fit method of each estimator.
            Note: will be split automatically for each fold.

        Returns
        -------
        self : object
            Fitted StackingTransformer instance.
        """
        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # Check input data
        # ---------------------------------------------------------------------
        # Check X and y
        # ``check_estimator`` does not allow ``force_all_finite=False``
        X, y = check_X_y(X, y,
                         accept_sparse=['csr'],  # allow csr, cast all others to csr
                         force_all_finite=True,  # do not allow  nan and inf
                         multi_output=False)  # allow only one column in y_train

        # Check X and sample_weight
        # X is alredy checked, but we need it to compare length of sample_weight
        if sample_weight is not None:
            X, sample_weight = check_X_y(X, sample_weight,
                                         accept_sparse=['csr'],
                                         force_all_finite=True,
                                         multi_output=False)

        # ---------------------------------------------------------------------
        # Check ``estimators``
        # ---------------------------------------------------------------------
        if self.estimators is None:
            if self.regression:
                self.estimators_ = [('dumregr', DummyRegressor(strategy='constant', constant=5.5))]
            else:
                self.estimators_ = [('dumclf', DummyClassifier(strategy='constant', constant=1))]
            # warnings.warn('No estimators were specified. '
            #               'Using single dummy estimator as demo.', UserWarning)
        else:
            if 0 == len(self.estimators):
                raise ValueError('List of estimators is empty')
            else:
                # Clone
                self.estimators_ = [(name, clone(estim)) for name, estim in self.estimators]
                # Check names of estimators
                names, estims = zip(*self.estimators_)
                self._validate_names(names)
                # Check if all estimators support ``sample_weight``
                if sample_weight is not None:
                    for name, estim in self.estimators_:
                        if not has_fit_parameter(estim, 'sample_weight'):
                            raise ValueError('Underlying estimator [%s] does not '
                                             'support sample weights.' % name)

        # ---------------------------------------------------------------------
        # Check other StackingTransformer parameters
        # ---------------------------------------------------------------------

        # ``variant``
        if self.variant not in ['A', 'B']:
            raise ValueError('Parameter ``variant`` must be set properly')

        # ``n_folds``
        if not isinstance(self.n_folds, int):
            raise ValueError('Parameter ``n_folds`` must be integer')
        if not self.n_folds > 1:
            raise ValueError('Parameter ``n_folds`` must be not less than 2')

        # ``verbose``
        if self.verbose not in [0, 1, 2]:
            raise ValueError('Parameter ``verbose`` must be 0, 1, or 2')

        # Additional check for inapplicable parameter combinations
        # If ``regression=True`` we ignore classification-specific
        # parameters and issue user warning
        if self.regression and (self.needs_proba or self.stratified):
            warn_str = ('This is regression task hence classification-specific'
                        'parameters set to ``True`` were ignored:')
            if self.needs_proba:
                self.needs_proba = False
                warn_str += ' ``needs_proba``'
            if self.stratified:
                self.stratified = False
                warn_str += ' ``stratified``'
            warnings.warn(warn_str, UserWarning)

        # ---------------------------------------------------------------------
        # Compute attributes (basic properties of data, number of estimators, etc.)
        # ---------------------------------------------------------------------
        self.train_shape_ = X.shape
        self.n_train_examples_ = X.shape[0]
        self.n_features_ = X.shape[1]
        if not self.regression:
            self.n_classes_ = len(np.unique(y))
        else:
            self.n_classes_ = None
        self.n_estimators_ = len(self.estimators_)
        self.train_footprint_ = self._get_footprint(X)

        # ---------------------------------------------------------------------
        # Specify default metric
        # ---------------------------------------------------------------------
        if self.metric is None and self.regression:
            self.metric_ = mean_absolute_error
        elif self.metric is None and not self.regression:
            if self.needs_proba:
                self.metric_ = log_loss
            else:
                self.metric_ = accuracy_score
        else:
            self.metric_ = self.metric
        # ---------------------------------------------------------------------
        # Create report header strings and print report header
        # ---------------------------------------------------------------------
        if self.verbose > 0:
            if self.regression:
                task_str = 'task:         [regression]'
            else:
                task_str = 'task:         [classification]'
                n_classes_str = 'n_classes:    [%d]' % self.n_classes_
            metric_str = 'metric:       [%s]' % self.metric_.__name__
            variant_str = 'variant:      [%s]' % self.variant
            n_estimators_str = 'n_estimators: [%d]' % self.n_estimators_

            print(task_str)
            if not self.regression:
                print(n_classes_str)
            print(metric_str)
            print(variant_str)
            print(n_estimators_str + '\n')
        # ---------------------------------------------------------------------
        # Initialize cross-validation split
        # Stratified can be used only for classification
        # ---------------------------------------------------------------------
        if not self.regression and self.stratified:
            self.kf_ = StratifiedKFold(n_splits=self.n_folds,
                                       shuffle=self.shuffle,
                                       random_state=self.random_state)
            # Save target to be able to create stratified split in ``transform`` method
            # This is more efficient than to save split indices
            self._y_ = y.copy()
        else:
            self.kf_ = KFold(n_splits=self.n_folds,
                             shuffle=self.shuffle,
                             random_state=self.random_state)
            self._y_ = None

        # ---------------------------------------------------------------------
        # Compute implicit number of classes to create appropriate empty arrays.
        # !!! Important. In order to unify array creation
        # variable ``n_classes_implicit_`` is always equal to 1, except the case
        # when we performing classification task with ``needs_proba=True``
        # ---------------------------------------------------------------------
        if not self.regression and self.needs_proba:
            self.n_classes_implicit_ = len(np.unique(y))
            self.action_ = 'predict_proba'
        else:
            self.n_classes_implicit_ = 1
            self.action_ = 'predict'

        # ---------------------------------------------------------------------
        # Create empty numpy array for train predictions (OOF)
        # !!! Important. We have to implicitly predict during fit
        # in order to compute CV scores, because
        # the most reasonable place to print out CV scores is fit method
        # ---------------------------------------------------------------------
        S_train = np.zeros((X.shape[0], self.n_estimators_ * self.n_classes_implicit_))

        # ---------------------------------------------------------------------
        # Prepare (clone) estmators for fitting and storing
        # We need models_A_ for both variant A and varian B
        # We need models_B_ for varian B only (in variant A attribute models_B_ is None)
        # ---------------------------------------------------------------------

        self.models_A_ = []
        self.models_B_ = None

        for n, est in self.estimators_:
            self.models_A_.append([clone(est) for _ in range(self.n_folds)])

        if self.variant in ['B']:
            self.models_B_ = [clone(est) for n, est in self.estimators_]

        # ---------------------------------------------------------------------
        # Create empty numpy array to store scores for each estimator and each fold
        # ---------------------------------------------------------------------
        self.scores_ = np.zeros((self.n_estimators_, self.n_folds))

        # ---------------------------------------------------------------------
        # Create empty list to store name, mean and std for each estimator
        # ---------------------------------------------------------------------
        self.mean_std_ = []

        # ---------------------------------------------------------------------
        # MAIN FIT PROCEDURE
        # ---------------------------------------------------------------------
        # Loop across estimators
        # ---------------------------------------------------------------------
        for estimator_counter, (name, estimator) in enumerate(self.estimators_):
            if self.verbose > 0:
                estimator_str = 'estimator %2d: [%s: %s]' % (estimator_counter, name, estimator.__class__.__name__)
                print(estimator_str)

            # -----------------------------------------------------------------
            # Loop across folds
            # -----------------------------------------------------------------
            for fold_counter, (tr_index, te_index) in enumerate(self.kf_.split(X, y)):
                # Split data and target
                X_tr = X[tr_index]
                y_tr = y[tr_index]
                X_te = X[te_index]
                y_te = y[te_index]

                # Split sample weights accordingly (if passed)
                if sample_weight is not None:
                    sample_weight_tr = sample_weight[tr_index]
                    # sample_weight_te = sample_weight[te_index]
                else:
                    sample_weight_tr = None
                    # sample_weight_te = None

                # Fit estimator
                _ = self._estimator_action(self.models_A_[estimator_counter][fold_counter],
                                           X_tr, y_tr, None,
                                           sample_weight=sample_weight_tr,
                                           action='fit',
                                           transform=self.transform_target)

                # Predict out-of-fold part of train set
                if 'predict_proba' == self.action_:
                    col_slice_estimator = slice(estimator_counter * self.n_classes_implicit_,
                                                estimator_counter * self.n_classes_implicit_ + self.n_classes_implicit_)
                else:
                    col_slice_estimator = estimator_counter
                S_train[te_index, col_slice_estimator] = self._estimator_action(self.models_A_[estimator_counter][fold_counter],
                                                                                None, None,
                                                                                X_te, action=self.action_,
                                                                                transform=self.transform_pred)
                # Compute score
                score = self.metric_(y_te, S_train[te_index, col_slice_estimator])
                self.scores_[estimator_counter, fold_counter] = score

                # Print fold score
                if self.verbose > 1:
                    fold_str = '    fold %2d:  [%.8f]' % (fold_counter, score)
                    print(fold_str)

            # Compute mean and std and save in dict
            estim_name = self.estimators_[estimator_counter][0]
            estim_mean = np.mean(self.scores_[estimator_counter])
            estim_std = np.std(self.scores_[estimator_counter])
            self.mean_std_.append((estim_name, estim_mean, estim_std))

            if self.verbose > 1:
                sep_str = '    ----'
                print(sep_str)

            # Compute mean + std (and full)
            if self.verbose > 0:
                mean_str = '    MEAN:     [%.8f] + [%.8f]\n' % (estim_mean, estim_std)
                print(mean_str)

            # Fit estimator on full train set
            if self.variant in ['B']:
                if self.verbose > 0:
                    print('    Fitting on full train set...\n')
                _ = self._estimator_action(self.models_B_[estimator_counter],
                                           X, y, None,
                                           sample_weight=sample_weight,
                                           action='fit',
                                           transform=self.transform_target)

        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------

        # Return fitted StackingTransformer instance
        return self

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def fit_transform(self, X, y, sample_weight=None):
        """Fit all base estimators and transform (predict) train set.

        Parameters
        ----------
        See docs for ``fit`` and ``transform`` methods.

        Returns
        -------
        X_transformed : 2d numpy array of shape [n_samples, n_estimators] or
                        [n_samples, n_estimators * n_classes]
            Out-of-fold predictions (OOF) for train set.
            This is stacked features for next level.
        """
        # ---------------------------------------------------------------------
        # All validation and procedures are done inside corresponding methods
        # fit and transform
        # ---------------------------------------------------------------------

        return self.fit(X, y, sample_weight).transform(X)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def transform(self, X, is_train_set=None):
        """Transform (predict) given data set.
        If ``X`` is train set:
            for each estimator return out-of-fold predictions (OOF).
        If ``X`` is any other set:
            variant A: for each estimator return mean (mode) of predictions
                made in each fold
            variant B: for each estimator return single prediction

        Parameters
        ----------
        X : 2d numpy array or sparse matrix of shape [n_samples, n_features]
            Input data

        is_train_set : boolean, default None
            Fallback parameter. In general case
                should not be used (should be None).
            Gives ability to explicitly specify that given dataset
                is train set or other set.

        Returns
        -------
        X_transformed : 2d numpy array of shape [n_samples, n_estimators] or
                        [n_samples, n_estimators * n_classes]
            Out-of-fold predictions (OOF) for train set.
            Regular or bagged predictions for any other set.
            This is stacked features for next level.
        """
        # Check if fitted
        check_is_fitted(self, ['models_A_'])

        # Input validation
        # ``check_estimator`` does not allow ``force_all_finite=False``
        X = check_array(X, accept_sparse=['csr'], force_all_finite=True)

        # *********************************************************************
        # Fitted StackingTransformer instance is bound to train set used for fitting.
        # So during transformation we have different actions for train set
        # and all other sets
        # *********************************************************************

        if is_train_set is None:
            is_train_set = self._check_identity(X)

        # Print
        if self.verbose > 0:
            if is_train_set:
                print('Train set was detected.')
            print('Transforming...\n')

        # *********************************************************************
        # Transform train set
        # *********************************************************************
        if is_train_set:

            # In case if user directly tells that it is train set but shape is different
            if self.train_shape_ != X.shape:
                raise ValueError('Train set must have the same shape '
                                 'in order to be transformed.')

            # Create empty numpy array for train predictions (OOF)
            S_train = np.zeros((X.shape[0], self.n_estimators_ * self.n_classes_implicit_))

            # -----------------------------------------------------------------
            # MAIN TRANSFORM (PREDICT) PROCEDURE for train set
            # -----------------------------------------------------------------
            # Loop across estimators
            # -----------------------------------------------------------------
            for estimator_counter, (name, estimator) in enumerate(self.estimators_):
                if self.verbose > 0:
                    estimator_str = 'estimator %2d: [%s: %s]' % (estimator_counter, name, estimator.__class__.__name__)
                    print(estimator_str)

                # -------------------------------------------------------------
                # Loop across folds
                # -------------------------------------------------------------
                for fold_counter, (tr_index, te_index) in enumerate(self.kf_.split(X, self._y_)):
                    # Split data
                    # X_tr = X[tr_index]
                    X_te = X[te_index]

                    # Predict out-of-fold part of train set
                    if 'predict_proba' == self.action_:
                        col_slice_estimator = slice(estimator_counter * self.n_classes_implicit_,
                                                    estimator_counter * self.n_classes_implicit_ + self.n_classes_implicit_)
                    else:
                        col_slice_estimator = estimator_counter
                    S_train[te_index, col_slice_estimator] = self._estimator_action(self.models_A_[estimator_counter][fold_counter],
                                                                                    None, None,
                                                                                    X_te, action=self.action_,
                                                                                    transform=self.transform_pred)
                    # Print
                    if self.verbose > 1:
                        fold_str = '    model from fold %2d: done' % fold_counter
                        print(fold_str)

                if self.verbose > 1:
                    sep_str = '    ----'
                    print(sep_str)

                if self.verbose > 0:
                    done_str = '    DONE\n'
                    print(done_str)

            # -----------------------------------------------------------------
            # Cast class labels to int
            # -----------------------------------------------------------------
            if not self.regression and not self.needs_proba:
                S_train = S_train.astype(int)

            # Return transformed data (OOF)
            return S_train  # X_transformed

        # *********************************************************************
        # Transform any other set
        # *********************************************************************
        else:
            # Check n_features
            if X.shape[1] != self.n_features_:
                raise ValueError('Inconsistent number of features.')

            # Create empty numpy array for test predictions
            S_test = np.zeros((X.shape[0], self.n_estimators_ * self.n_classes_implicit_))

            # ---------------------------------------------------------------------
            # MAIN TRANSFORM (PREDICT) PROCEDURE for any other set
            # -----------------------------------------------------------------
            # Loop across estimators
            # -----------------------------------------------------------------
            for estimator_counter, (name, estimator) in enumerate(self.estimators_):
                if self.verbose > 0:
                    estimator_str = 'estimator %2d: [%s: %s]' % (estimator_counter, name, estimator.__class__.__name__)
                    print(estimator_str)
                # -------------------------------------------------------------
                # Variant A
                # -------------------------------------------------------------
                if self.variant in ['A']:
                    # Create empty numpy array, which will contain temporary predictions
                    # for test set made in each fold
                    S_test_temp = np.zeros((X.shape[0], self.n_folds * self.n_classes_implicit_))
                    # ---------------------------------------------------------
                    # Loop across fitted models (it is the same as loop across folds)
                    # ---------------------------------------------------------
                    for fold_counter, model in enumerate(self.models_A_[estimator_counter]):
                        # Predict test set in each fold
                        if 'predict_proba' == self.action_:
                            col_slice_fold = slice(fold_counter * self.n_classes_implicit_,
                                                   fold_counter * self.n_classes_implicit_ + self.n_classes_implicit_)
                        else:
                            col_slice_fold = fold_counter
                        S_test_temp[:, col_slice_fold] = self._estimator_action(model, None, None, X,
                                                                                action=self.action_,
                                                                                transform=self.transform_pred)
                        # Print
                        if self.verbose > 1:
                            fold_str = '    model from fold %2d: done' % fold_counter
                            print(fold_str)

                    if self.verbose > 1:
                        sep_str = '    ----'
                        print(sep_str)

                    # ---------------------------------------------------------
                    # Compute mean or mode (majority voting) of predictions for test set
                    # ---------------------------------------------------------
                    if 'predict_proba' == self.action_:
                        # Here we copute means of probabilirties for each class
                        for class_id in range(self.n_classes_implicit_):
                            S_test[:, estimator_counter * self.n_classes_implicit_ + class_id] = np.mean(S_test_temp[:, class_id::self.n_classes_implicit_], axis=1)
                    else:
                        if self.regression:
                            S_test[:, estimator_counter] = np.mean(S_test_temp, axis=1)
                        else:
                            S_test[:, estimator_counter] = st.mode(S_test_temp, axis=1)[0].ravel()

                    if self.verbose > 0:
                        done_str = '    DONE\n'
                        print(done_str)

                # -------------------------------------------------------------
                # Variant B
                # -------------------------------------------------------------
                else:
                    if 'predict_proba' == self.action_:
                        col_slice_estimator = slice(estimator_counter * self.n_classes_implicit_,
                                                    estimator_counter * self.n_classes_implicit_ + self.n_classes_implicit_)
                    else:
                        col_slice_estimator = estimator_counter
                    S_test[:, col_slice_estimator] = self._estimator_action(self.models_B_[estimator_counter],
                                                                            None, None, X,
                                                                            action=self.action_,
                                                                            transform=self.transform_pred)

                    if self.verbose > 0:
                        done_str = '    DONE\n'
                        print(done_str)

            # ---------------------------------------------------------------------
            # Cast class labels to int
            # ---------------------------------------------------------------------
            if not self.regression and not self.needs_proba:
                S_test = S_test.astype(int)

            return S_test  # X_transformed

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def _transformer(self, y, func=None):
        """Transforms target variable and prediction
        """
        if func is None:
            return y
        else:
            return func(y)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def _estimator_action(self, estimator, X_train, y_train, X_test,
                          sample_weight=None, action=None,
                          transform=None):
        """Performs estimator action.
        This wrapper gives us ability to choose action dynamically
        (e.g. ``predict`` or ``predict_proba``).
        Note. Through ``_estimator_action`` and then through ``_transformer``
        we apply ``transform_target`` and ``transform_pred`` functions if
        given by user on the target and prediction in each fold separately
        to be able to calculate proper scores.
        """
        if 'fit' == action:
            # We use following condition, because some estimators (e.g. Lars)
            # may not have ``sample_weight`` parameter of ``fit`` method
            if sample_weight is not None:
                return estimator.fit(X_train, self._transformer(y_train, func=transform),
                                     sample_weight=sample_weight)
            else:
                return estimator.fit(X_train, self._transformer(y_train, func=transform))
        elif 'predict' == action:
            return self._transformer(estimator.predict(X_test), func=transform)
        elif 'predict_proba' == action:
            return self._transformer(estimator.predict_proba(X_test), func=transform)
        else:
            raise ValueError('Parameter action must be set properly')

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def _random_choice(self, n, size, bound=2**30):
        """
        Memory efficient substitute for np.random.choice without replacement

        Parameters:
        ===========
        n : int
            Upper value for range to chose from: [0, n).
            This parameter is bounded (see bound).
        size: int
            Number of values to chose
        bound : int
            Upper random int for backward compatibility
            with some older numpy versions

        Returns:
        ========
        ids : 1d numpy array of shape (size, ) dtype=np.int32
        """
        try:
            if n < size:
                raise ValueError('Drawing without replacement: '
                                 '``n`` cannot be less than ``size``')

            ids = []
            while len(ids) < size:
                rnd = np.random.randint(min(bound, n))
                if rnd not in ids:
                    ids.append(rnd)
            return np.array(ids, dtype=np.int32)

        except Exception:
            raise ValueError('Internal error. '
                             'Please save traceback and inform developers.')

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def _get_footprint(self, X, n_items=1000):
        """Selects ``n_items`` random elements from 2d numpy array or
        sparse matrix (or all elements if their number is less or equal
        to ``n_items``).
        """
        try:
            footprint = []
            r, c = X.shape
            n = r * c
            # np.random.seed(0) # for development

            # OOM with large arrays (see #29)
            # ids = np.random.choice(n, min(n_items, n), replace=False)

            ids = self._random_choice(n, min(n_items, n))

            for i in ids:
                row = i // c
                col = i - row * c
                footprint.append((row, col, X[row, col]))

            return footprint

        except Exception:
            raise ValueError('Internal error. '
                             'Please save traceback and inform developers.')

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def _check_identity(self, X,
                        rtol=1e-05, atol=1e-08,
                        equal_nan=False):
        """Checks 2d numpy array or sparse matrix identity
        by its shape and footprint.
        """
        try:
            # Check shape
            if X.shape != self.train_shape_:
                return False
            # Check footprint
            try:
                for coo in self.train_footprint_:
                    assert np.isclose(X[coo[0], coo[1]], coo[2], rtol=rtol, atol=atol, equal_nan=equal_nan)
                return True
            except AssertionError:
                return False

        except Exception:
            raise ValueError('Internal error. '
                             'Please save traceback and inform developers.')

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def _get_params(self, attr, deep=True):
        """Gives ability to get parameters of nested estimators
        """
        out = super(StackingTransformer, self).get_params(deep=False)
        if not deep:
            return out
        estimators = getattr(self, attr)
        if estimators is None:
            return out
        out.update(estimators)
        for name, estimator in estimators:
            for key, value in iter(estimator.get_params(deep=True).items()):
                out['%s__%s' % (name, key)] = value
        return out

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def get_params(self, deep=True):
        """Get parameters of StackingTransformer and base estimators.

        Parameters
        ----------
        deep : boolean
            If False - get parameters of StackingTransformer
            If True - get parameters of StackingTransformer and base estimators
        """
        return self._get_params('estimators', deep=deep)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def _validate_names(self, names):
        """Validates estimator names
        """
        if len(set(names)) != len(names):
            raise ValueError('Names provided are not unique: '
                             '%s' % list(names))
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError('Estimator names conflict with constructor '
                             'arguments: %s' % sorted(invalid_names))
        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got '
                             '%s' % invalid_names)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def is_train_set(self, X):
        """Checks if given data set was used to train
        StackingTransformer instance.

        Parameters
        ----------
        X : 2d numpy array or sparse matrix of shape [n_samples, n_features]
            Input data

        Returns
        -------
        check_result : boolean
            True - if X was used to train StackingTransformer instance
            False - otherwise
        """
        # Check if fitted
        check_is_fitted(self, ['models_A_'])
        # Input validation
        X = check_array(X, accept_sparse=['csr'], force_all_finite=True)
        return self._check_identity(X)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
