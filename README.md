[![PyPI version](https://img.shields.io/pypi/v/vecstack.svg?colorB=4cc61e)](https://pypi.python.org/pypi/vecstack)
[![PyPI license](https://img.shields.io/pypi/l/vecstack.svg)](https://github.com/vecxoz/vecstack/blob/master/LICENSE.txt)
[![Build Status](https://travis-ci.org/vecxoz/vecstack.svg?branch=master)](https://travis-ci.org/vecxoz/vecstack)
[![Coverage Status](https://coveralls.io/repos/github/vecxoz/vecstack/badge.svg?branch=master)](https://coveralls.io/github/vecxoz/vecstack?branch=master)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/vecstack.svg)](https://pypi.python.org/pypi/vecstack/)

# vecstack
Python package for stacking (stacked generalization) featuring lightweight ***functional API*** and fully compatible ***scikit-learn API***  
Convenient way to automate OOF computation, prediction and bagging using any number of models  

* [Functional API](https://github.com/vecxoz/vecstack#usage-functional-api):
    * Minimalistic. Get your stacked features in a single line
    * RAM-friendly. The lowest possible memory consumption
    * Kaggle-ready. Stacked features and hyperparameters from each run can be [automatically saved](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L209) in files. No more mess at the end of the competition.  [Log example](https://github.com/vecxoz/vecstack/blob/master/examples/03_log_example.txt)
* [Scikit-learn API](https://github.com/vecxoz/vecstack#usage-scikit-learn-api):
    * Standardized. Fully scikit-learn compatible transformer class exposing `fit` and `transform` methods
    * Pipeline-certified. Implement and deploy [multilevel stacking](https://github.com/vecxoz/vecstack/blob/master/examples/04_sklearn_api_regression_pipeline.ipynb) like it's no big deal using `sklearn.pipeline.Pipeline` 
    * And of course `FeatureUnion` is also invited to the party
* Overall specs:
    * Use any sklearn-like estimators
    * Perform [classification and regression](https://github.com/vecxoz/vecstack/blob/master/vecstack/coresk.py#L83) tasks
    * Predict [class labels or probabilities](https://github.com/vecxoz/vecstack/blob/master/vecstack/coresk.py#L119) in classification task
    * Apply any [user-defined metric](https://github.com/vecxoz/vecstack/blob/master/vecstack/coresk.py#L124)
    * Apply any [user-defined transformations](https://github.com/vecxoz/vecstack/blob/master/vecstack/coresk.py#L87) for target and prediction
    * Python 3.5 and higher, [unofficial support for Python 2.7 and 3.4](https://github.com/vecxoz/vecstack/blob/master/PY2.md)
    * Win, Linux, Mac
    * [MIT license](https://github.com/vecxoz/vecstack/blob/master/LICENSE.txt)
    * Depends on **numpy**, **scipy**, **scikit-learn>=0.18**

# Get started
* [FAQ](https://github.com/vecxoz/vecstack#stacking-faq)
* [Installation guide](https://github.com/vecxoz/vecstack#installation)
* Usage:
    * [Functional API](https://github.com/vecxoz/vecstack#usage-functional-api)
    * [Scikit-learn API](https://github.com/vecxoz/vecstack#usage-scikit-learn-api)
* Tutorials:
    * [Stacking concept + Pictures + Stacking implementation from scratch](https://github.com/vecxoz/vecstack/blob/master/examples/00_stacking_concept_pictures_code.ipynb)
* Examples (all examples are valid for both API with little [difference in parameters](https://github.com/vecxoz/vecstack#21-how-do-parameters-of-stacking-function-and-stackingtransformer-correspond)):
    * Functional API:
        * [Regression](https://github.com/vecxoz/vecstack/blob/master/examples/01_regression.ipynb)
        * [Classification with class labels](https://github.com/vecxoz/vecstack/blob/master/examples/02_classification_with_class_labels.ipynb)
        * [Classification with probabilities + Detailed workflow](https://github.com/vecxoz/vecstack/blob/master/examples/03_classification_with_proba_detailed_workflow.ipynb)
    * Scikit-learn API:
        * [Regression + Multilevel stacking using Pipeline](https://github.com/vecxoz/vecstack/blob/master/examples/04_sklearn_api_regression_pipeline.ipynb)
* Documentation:
    * [Functional API](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py#L133) or type ```>>> help(stacking)```
    * [Scikit-learn API](https://github.com/vecxoz/vecstack/blob/master/vecstack/coresk.py#L64) or type ```>>> help(StackingTransformer)```

# Installation

***Note:*** Python 3.5 or higher is required. If you’re still using Python 2.7 or 3.4 see [installation details here](https://github.com/vecxoz/vecstack/blob/master/PY2.md)  

* ***Classic 1st time installation (recommended):*** 
    * `pip install vecstack`
* Install for current user only (if you have some troubles with write permission):
    * `pip install --user vecstack`
* If your PATH doesn't work: 
    * `/usr/bin/python -m pip install vecstack`
    * `C:/Python36/python -m pip install vecstack`
* Upgrade vecstack and all dependencies:
    * `pip install --upgrade vecstack`
* Upgrade vecstack WITHOUT upgrading dependencies:
    * `pip install --upgrade --no-deps vecstack`
* Upgrade directly from GitHub WITHOUT upgrading dependencies:
    * `pip install --upgrade --no-deps https://github.com/vecxoz/vecstack/archive/master.zip`
* Uninstall
    * `pip uninstall vecstack`

# Usage. Functional API
```python
from vecstack import stacking

# Get your data

# Initialize 1st level estimators
models = [LinearRegression(),
          Ridge(random_state=0)]

# Get your stacked features in a single line
S_train, S_test = stacking(models, X_train, y_train, X_test, regression=True, verbose=2)

# Use 2nd level estimator with stacked features
```

# Usage. Scikit-learn API
```python
from vecstack import StackingTransformer

# Get your data

# Initialize 1st level estimators
estimators = [('lr', LinearRegression()),
              ('ridge', Ridge(random_state=0))]
              
# Initialize StackingTransformer
stack = StackingTransformer(estimators, regression=True, verbose=2)

# Fit
stack = stack.fit(X_train, y_train)

# Get your stacked features
S_train = stack.transform(X_train)
S_test = stack.transform(X_test)

# Use 2nd level estimator with stacked features
```

# Stacking FAQ

1.  [How can I report an issue? How can I ask a question about stacking or vecstack package?](https://github.com/vecxoz/vecstack#1-how-can-i-report-an-issue-how-can-i-ask-a-question-about-stacking-or-vecstack-package)
2.  [How can I say thanks?](https://github.com/vecxoz/vecstack#2-how-can-i-say-thanks)
3.  [How to cite vecstack?](https://github.com/vecxoz/vecstack#3-how-to-cite-vecstack)
4.  [What is stacking?](https://github.com/vecxoz/vecstack#4-what-is-stacking)
5.  [What about stacking name?](https://github.com/vecxoz/vecstack#5-what-about-stacking-name)
6.  [Do I need stacking at all?](https://github.com/vecxoz/vecstack#6-do-i-need-stacking-at-all)
7.  [Can you explain stacking (stacked generalization) in 10 lines of code?](https://github.com/vecxoz/vecstack#7-can-you-explain-stacking-stacked-generalization-in-10-lines-of-code)
8.  [Why do I need complicated inner procedure for stacking?](https://github.com/vecxoz/vecstack#8-why-do-i-need-complicated-inner-procedure-for-stacking)
9.  [I want to implement stacking (stacked generalization) from scratch. Can you help me?](https://github.com/vecxoz/vecstack#9-i-want-to-implement-stacking-stacked-generalization-from-scratch-can-you-help-me)
10. [What is OOF?](https://github.com/vecxoz/vecstack#10-what-is-oof)
11. [What are *estimator*, *learner*, *model*?](https://github.com/vecxoz/vecstack#11-what-are-estimator-learner-model)
12. [What is *blending*? How is it related to stacking?](https://github.com/vecxoz/vecstack#12-what-is-blending-how-is-it-related-to-stacking)
13. [How to optimize weights for weighted average?](https://github.com/vecxoz/vecstack#13-how-to-optimize-weights-for-weighted-average)
14. [What is better: weighted average for current level or additional level?](https://github.com/vecxoz/vecstack#14-what-is-better-weighted-average-for-current-level-or-additional-level)
15. [What is *bagging*? How is it related to stacking?](https://github.com/vecxoz/vecstack#15-what-is-bagging-how-is-it-related-to-stacking)
16. [How many models should I use on a given stacking level?](https://github.com/vecxoz/vecstack#16-how-many-models-should-i-use-on-a-given-stacking-level)
17. [How many stacking levels should I use?](https://github.com/vecxoz/vecstack#17-how-many-stacking-levels-should-i-use)
18. [How do I choose models for stacking?](https://github.com/vecxoz/vecstack#18-how-do-i-choose-models-for-stacking)
19. [I am trying hard but still can't beat my best single model with stacking. What is wrong?](https://github.com/vecxoz/vecstack#19-i-am-trying-hard-but-still-cant-beat-my-best-single-model-with-stacking-what-is-wrong)
20. [What should I choose: functional API (`stacking` function) or Scikit-learn API (`StackingTransformer`)?](https://github.com/vecxoz/vecstack#20-what-should-i-choose-functional-api-stacking-function-or-scikit-learn-api-stackingtransformer)
21. [How do parameters of `stacking` function and `StackingTransformer` correspond?](https://github.com/vecxoz/vecstack#21-how-do-parameters-of-stacking-function-and-stackingtransformer-correspond)
22. [Why Scikit-learn API was implemented as transformer and not predictor?](https://github.com/vecxoz/vecstack#22-why-scikit-learn-api-was-implemented-as-transformer-and-not-predictor)
23. [How to estimate stacking training time and number of models which will be built?](https://github.com/vecxoz/vecstack#23-how-to-estimate-stacking-training-time-and-number-of-models-which-will-be-built)
24. [Which stacking variant should I use: 'A' ('oof_pred_bag') or 'B' ('oof_pred')?](https://github.com/vecxoz/vecstack#24-which-stacking-variant-should-i-use-a-oof_pred_bag-or-b-oof_pred)
25. [How to choose number of folds?](https://github.com/vecxoz/vecstack#25-how-to-choose-number-of-folds)
26. [When I transform train set I see 'Train set was detected'. What does it mean?](https://github.com/vecxoz/vecstack#26-when-i-transform-train-set-i-see-train-set-was-detected-what-does-it-mean)
27. [How is the very first stacking level called: L0 or L1? Where does counting start?](https://github.com/vecxoz/vecstack#27-how-is-the-very-first-stacking-level-called-l0-or-l1-where-does-counting-start)
28. [Can I use `(Randomized)GridSearchCV` to tune the whole stacking Pipeline?](https://github.com/vecxoz/vecstack#28-can-i-use-randomizedgridsearchcv-to-tune-the-whole-stacking-pipeline)
29. [How to define custom metric, especially AUC?](https://github.com/vecxoz/vecstack#29-how-to-define-custom-metric-especially-auc)
30. [Do folds (splits) have to be the same across estimators and stacking levels? How does `random_state` work?](https://github.com/vecxoz/vecstack#30-do-folds-splits-have-to-be-the-same-across-estimators-and-stacking-levels-how-does-random_state-work)

### 1. How can I report an issue? How can I ask a question about stacking or vecstack package?

Just open an issue [here](https://github.com/vecxoz/vecstack/issues).  
Ask me anything on the topic.  
I'm a bit busy, so typically I answer on the next day.  

### 2. How can I say thanks?

Just give me a star in the top right corner of the repository page.  

### 3. How to cite vecstack?

```
@misc{vecstack2016,
       author = {Igor Ivanov},
       title = {Vecstack},
       year = {2016},
       publisher = {GitHub},
       howpublished = {\url{https://github.com/vecxoz/vecstack}},
}
```

### 4. What is stacking?

Stacking (stacked generalization) is a machine learning ensembling technique.  
Main idea is to use predictions as features.  
More specifically we predict train set (in CV-like fashion) and test set using some 1st level model(s), and then use these predictions as features for 2nd level model. You can find more details (concept, pictures, code) in [stacking tutorial](https://github.com/vecxoz/vecstack/blob/master/examples/00_stacking_concept_pictures_code.ipynb).  
Also make sure to check out: 
* [Ensemble Learning](https://en.wikipedia.org/wiki/Ensemble_learning) ([Stacking](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking)) in Wikipedia
* Classical [Kaggle Ensembling Guide](https://mlwave.com/kaggle-ensembling-guide/)
* [Stacked Generalization](https://www.researchgate.net/publication/222467943_Stacked_Generalization) paper by David H. Wolpert
    
### 5. What about stacking name?

Often it is also called *stacked generalization*. The term is derived from the verb *to stack* (to put together, to put on top of each other). It implies that we put some models on top of other models, i.e. train some models on predictions of other models. From another point of view we can say that we stack predictions in order to use them as features.  

### 6. Do I need stacking at all?

It depends on specific business case. The main thing to know about stacking is that it requires ***significant computing resources***. [No Free Lunch Theorem](https://en.wikipedia.org/wiki/There_ain%27t_no_such_thing_as_a_free_lunch) applies as always. Stacking can give you an improvement but for certain price (deployment, computation, maintenance). Only experiment for given business case will give you an answer: is it worth an effort and money.  

At current point large part of stacking users are participants of machine learning competitions. On Kaggle you can't go too far without ensembling. I can secretly tell you that at least top half of leaderboard in pretty much any competition uses ensembling (stacking) in some way. Stacking is less popular in production due to time and resource constraints, but I think it gains popularity.  
   
### 7. Can you explain stacking (stacked generalization) in 10 lines of code?

[Of course](https://github.com/vecxoz/vecstack/blob/master/examples/00_stacking_concept_pictures_code.ipynb)
    
### 8. Why do I need complicated inner procedure for stacking?

I can just do the following. Why not?  

```python
model_L1 = XGBRegressor()
model_L1 = model_L1.fit(X_train, y_train)
S_train = model_L1.predict(X_train).reshape(-1, 1)  # <- DOES NOT work due to overfitting. Must be CV
S_test = model_L1.predict(X_test).reshape(-1, 1)
model_L2 = LinearRegression()
model_L2 = model_L2.fit(S_train, y_train)
final_prediction = model_L2.predict(S_test)
```

Code above will give meaningless result. If we fit on `X_train` we can’t just predict `X_train`, because our 1st level model has already seen `X_train`, and its prediction will be overfitted. To avoid overfitting we perform cross-validation procedure and in each fold we predict out-of-fold (OOF) part of `X_train`. You can find more details (concept, pictures, code) in [stacking tutorial](https://github.com/vecxoz/vecstack/blob/master/examples/00_stacking_concept_pictures_code.ipynb).  

### 9. I want to implement stacking (stacked generalization) from scratch. Can you help me?

[Not a problem](https://github.com/vecxoz/vecstack/blob/master/examples/00_stacking_concept_pictures_code.ipynb)  
    
### 10. What is OOF?

OOF is abbreviation for out-of-fold prediction. It's also known as *OOF features*, *stacked features*, *stacking features*, etc. Basically it means predictions for the part of train data that model haven't seen during training.  
    
### 11. What are *estimator*, *learner*, *model*?

Basically it is the same thing meaning *machine learning algorithm*. Often these terms are used interchangeably.  
Speaking about inner stacking mechanics, you should remember that when you have *single 1st level model* there will be at least `n_folds` separate models *trained in each CV fold* on different subsets of data. See [Q23](https://github.com/vecxoz/vecstack#23-how-to-estimate-stacking-training-time-and-number-of-models-which-will-be-built) for more details.  

### 12. What is *blending*? How is it related to stacking?

Basically it is the same thing. Both approaches use predictions as features.  
Often these terms are used interchangeably.  
The difference is how we generate features (predictions) for the next level:  
* *stacking*: perform cross-validation procedure and predict each part of train set (OOF)
* *blending*: predict fixed holdout set

*vecstack* package supports only *stacking* i.e. cross-validation approach. For given `random_state` value (e.g. 42) folds (splits) will be the same across all estimators. See also [Q30](https://github.com/vecxoz/vecstack#30-do-folds-splits-have-to-be-the-same-across-estimators-and-stacking-levels-how-does-random_state-work).

### 13. How to optimize weights for weighted average?
    
You can use for example:

* `scipy.optimize.minimize`
* `scipy.optimize.differential_evolution`

### 14. What is better: weighted average for current level or additional level?

By default you can start from weighted average. It is easier to apply and more chances that it will give good result. Then you can try additional level which potentially can outperform weighted average (but not always and not in an easy way). Experiment is your friend.  

### 15. What is *bagging*? How is it related to stacking?

[Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating) or Bootstrap aggregating works as follows: generate subsets of training set, train models on these subsets and then find average of predictions.  
Also term *bagging* is often used to describe following approaches:
* train several different models on the same data and average predictions
* train same model with different random seeds on the same data and average predictions

So if we run stacking and just average predictions - it is *bagging*.
    
### 16. How many models should I use on a given stacking level?

***Note 1:*** The best architecture can be found only by experiment.  
***Note 2:*** Always remember that higher number of levels or models does NOT guarantee better result. The key to success in stacking (and ensembling in general) is diversity - low correlation between models.  

It depends on many factors like type of problem, type of data, quality of models, correlation of models, expected result, etc.  
Some example configurations are listed below.  
* Reasonable starting point:  
    * `L1: 2-10 models -> L2: weighted (rank) average or single model`
* Then try to add more 1st level models and additional level:  
    * `L1: 10-50 models -> L2: 2-10 models -> L3: weighted (rank) average`
* If you're crunching numbers at Kaggle and decided to go wild:  
    * `L1: 100-inf models -> L2: 10-50 models -> L3: 2-10 models -> L4: weighted (rank) average`

You can also find some winning stacking architectures on [Kaggle blog](http://blog.kaggle.com/), e.g.: [1st place in Homesite Quote Conversion](http://blog.kaggle.com/2016/04/08/homesite-quote-conversion-winners-write-up-1st-place-kazanova-faron-clobber/).  
    
### 17. How many stacking levels should I use?

***Note 1:*** The best architecture can be found only by experiment.  
***Note 2:*** Always remember that higher number of levels or models does NOT guarantee better result. The key to success in stacking (and ensembling in general) is diversity - low correlation between models.  

For some example configurations see [Q16](https://github.com/vecxoz/vecstack#16-how-many-models-should-i-use-on-a-given-stacking-level).  

### 18. How do I choose models for stacking?

Based on experiments and correlation (e.g. Pearson). Less correlated models give better result. It means that we should never judge our models by accuracy only. We should also consider correlation (how given model is different from others). Sometimes inaccurate but very different model can add substantial value to resulting ensemble.  

### 19. I am trying hard but still can't beat my best single model with stacking. What is wrong?

Nothing is wrong. Stacking is advanced complicated technique. It's hard to make it work. ***Solution:*** make sure to try weighted (rank) average first instead of additional level with some advanced models. Average is much easier to apply and in most cases it will surely outperform your best model. If still no luck - then probably your models are highly correlated.  

### 20. What should I choose: functional API (`stacking` function) or Scikit-learn API (`StackingTransformer`)?

Quick guide:

* By default start from `StackingTransformer` with familiar scikit-learn interface and logic
* If you need low RAM consumption try `stacking` function but remember that it does not store models and does not have scikit-learn capabilities
    
Stacking API comparison:

| **Property**   | **stacking function** | **StackingTransformer** |
|----------------|:---------------------:|:-----------------------:|
| Execution time | Same                  | Same                    |
| RAM            | Consumes the ***smallest possible amount of RAM***. Does not store models. At any point in time only one model is alive. Logic: train model -> predict -> delete -> etc. When execution ends all RAM is released.| Consumes ***much more RAM***. It stores all models built in each fold. This price is paid for standard scikit-learn capabilities like `Pipeline` and `FeatureUnion`. |
| Access to models after training | No | Yes |
| Compatibility with `Pipeline` and `FeatureUnion` | No | Yes |
| Estimator implementation restrictions | Must have only `fit` and `predict` (`predict_proba`) methods | Must be fully scikit-learn compatible |
| `NaN` and `inf` in input data | Allowed | Not allowed |
| Can automatically save OOF and log in files | Yes | No |
| Input dimensionality (`X_train`, `X_test`) | Arbitrary | 2-D |
    
### 21. How do parameters of `stacking` function and `StackingTransformer` correspond?

| **stacking function**                 | **StackingTransformer**           |
|---------------------------------------|-----------------------------------|
| `models=[Ridge()]`                    | `estimators=[('ridge', Ridge())]` |
| `mode='oof_pred_bag'` (alias `'A'`)   | `variant='A'`                     |
| `mode='oof_pred'` (alias `'B'`)       | `variant='B'`                     |
    
### 22. Why Scikit-learn API was implemented as transformer and not predictor?

* By nature stacking procedure is predictor, but ***by application*** it is definitely transformer.
* Transformer architecture was chosen because first of all user needs direct access to OOF. I.e. the ability to compute correlations, weighted average, etc.
* If you need predictor based on `StackingTransformer` you can easily create it via `Pipeline` by adding on the top of `StackingTransformer` some regressor or classifier.
* Transformer makes it easy to create any number of stacking levels. Using Pipeline we can easily create multilevel stacking by just adding several `StackingTransformer`'s on top of each other and then some final regressor or classifier.  
    
### 23. How to estimate stacking training time and number of models which will be built?

***Note:*** Stacking usually takes long time. It's expected (inevitable) behavior.  

We can compute total number of models which will be built during stacking procedure using following formulas:
* Variant A: `n_models_total = n_estimators * n_folds`  
* Variant B: `n_models_total = n_estimators * n_folds + n_estimators`  

Let's look at example. Say we define our stacking procedure as follows:
```python
estimators_L1 = [('lr', LinearRegression()),
                 ('ridge', Ridge())]
stack = StackingTransformer(estimators_L1, n_folds=4)
```
So we have two 1st level estimators and 4 folds. It means stacking procedure will build the following number of models: 
* Variant A: 8 models total. Each model is trained on 3/4 of `X_train`.  
* Variant B: 10 models total. 8 models are trained on 3/4 of `X_train` and 2 models on full `X_train`.  

Compute time:
* If estimators have relatively *similar training time*, we can roughly compute total training time as: `time_total = n_models_total * time_of_one_model`
* If estimators have *different training time*, we should compute number of models and time for each estimator separately (set `n_estimators=1` in formulas above) and then sum up times.
    
### 24. Which stacking variant should I use: 'A' ('oof_pred_bag') or 'B' ('oof_pred')?
    
You can find out only by experiment. Default choice is variant ***A***, because it takes ***less time*** and there should be no significant difference in result. But of course you may also try variant B. For more details see [stacking tutorial](https://github.com/vecxoz/vecstack/blob/master/examples/00_stacking_concept_pictures_code.ipynb).  
    
### 25. How to choose number of folds?

***Note:*** Remember that higher number of folds substantially increase training time (and RAM consumption for StackingTransformer). See [Q23](https://github.com/vecxoz/vecstack#23-how-to-estimate-stacking-training-time-and-number-of-models-which-will-be-built).  

* Standard approach: 4 or 5 folds.
* If data is big: 3 folds.
* If data is small: you can try more folds like 10 or so.
    
### 26. When I transform train set I see 'Train set was detected'. What does it mean?

***Note 1:*** It is ***NOT allowed to change train set*** between calls to `fit` and `transform` methods. Due to stacking nature transformation is different for train set and any other set. If train set is changed after training, stacking procedure will not be able to correctly identify it and transformation will be wrong.  

***Note 2:*** To be correctly detected train set does not necessarily have to be identical (exactly the same). It must have the same shape and all values must be *close* (`np.isclose` is used for checking). So if you somehow regenerate your train set you should not worry about numerical precision.  

If you transform `X_train` and see 'Train set was detected' everything is OK. If you transform `X_train` but you don't see this message then something went wrong. Probably your train set was changed (it is not allowed). In this case you have to retrain `StackingTransformer`. For more details see [stacking tutorial](https://github.com/vecxoz/vecstack/blob/master/examples/00_stacking_concept_pictures_code.ipynb) or [Q8](https://github.com/vecxoz/vecstack#8-why-do-i-need-complicated-inner-procedure-for-stacking).  

### 27. How is the very first stacking level called: L0 or L1? Where does counting start?

***Common convention:*** The very first bunch of models which are trained on initial raw data are called ***L1***. On top of L1 we have so called *stacker level* or *meta level* or L2 i.e. models which are trained on predictions of L1 models. Count continues in the same fashion up to arbitrary number of levels.  

I use this convention in my code and docs. But of course your Kaggle teammates may use some other naming approach, so you should clarify this for your specific case.  

### 28. Can I use `(Randomized)GridSearchCV` to tune the whole stacking Pipeline?

Yes, technically you can, but it ***is not recommended*** because this approach will lead to redundant computations. General practical advice is to ***tune each estimator separately*** and then use tuned estimators on the 1st level of stacking. Higher level estimators should be tuned in the same fashion using OOF from previous level. For manual tuning you can use `stacking` function or `StackingTransformer` with a single 1st level estimator.  

### 29. How to define custom metric, especially AUC?

```python
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

def auc(y_true, y_pred):
    """ROC AUC metric for both binary and multiclass classification.
    
    Parameters
    ----------
    y_true : 1d numpy array
        True class labels
    y_pred : 2d numpy array
        Predicted probabilities for each class
    """
    ohe = OneHotEncoder(sparse=False)
    y_true = ohe.fit_transform(y_true.reshape(-1, 1))
    auc_score = roc_auc_score(y_true, y_pred)
    return auc_score
```

### 30. Do folds (splits) have to be the same across estimators and stacking levels? How does `random_state` work?

To ensure better result, folds (splits) have to be the same across all estimators and all stacking levels. It means that `random_state` has to be the same in every call to `stacking` function or `StackingTransformer`. This is default behavior of `stacking` function and `StackingTransformer` (by default `random_state=0`). If you want to try different folds (splits) try to set different `random_state` values.  


# Stacking concept

1. We want to predict train set and test set with some 1st level model(s), and then use these predictions as features for 2nd level model(s).  
2. Any model can be used as 1st level model or 2nd level model.
3. To avoid overfitting (for train set) we use cross-validation technique and in each fold we predict out-of-fold (OOF) part of train set.
4. The common practice is to use from 3 to 10 folds.
5. Predict test set:
   * **Variant A:** In each fold we predict test set, so after completion of all folds we need to find mean (mode) of all temporary test set predictions made in each fold. 
   * **Variant B:** We do not predict test set during cross-validation cycle. After completion of all folds we perform additional step: fit model on full train set and predict test set once. This approach takes more time because we need to perform one additional fitting.
6. As an example we look at stacking implemented with single 1st level model and 3-fold cross-validation.
7. Pictures:
   * **Variant A:** Three pictures describe three folds of cross-validation. After completion of all three folds we get single train feature and single test feature to use with 2nd level model. 
   * **Variant B:** First three pictures describe three folds of cross-validation (like in Variant A) to get single train feature and fourth picture describes additional step to get single test feature.
8. We can repeat this cycle using other 1st level models to get more features for 2nd level model.
9. You can also look at animation of [Variant A](https://github.com/vecxoz/vecstack#variant-a-animation) and [Variant B](https://github.com/vecxoz/vecstack#variant-b-animation).

# Variant A

![Fold 1 of 3](https://github.com/vecxoz/vecstack/raw/master/pic/dia1.png "Fold 1 of 3")
***
![Fold 2 of 3](https://github.com/vecxoz/vecstack/raw/master/pic/dia2.png "Fold 2 of 3")
***
![Fold 3 of 3](https://github.com/vecxoz/vecstack/raw/master/pic/dia3.png "Fold 3 of 3")

# Variant A. Animation

![Variant A. Animation](https://github.com/vecxoz/vecstack/raw/master/pic/animation1.gif "Variant A. Animation")

# Variant B

![Step 1 of 4](https://github.com/vecxoz/vecstack/raw/master/pic/dia4.png "Step 1 of 4")
***
![Step 2 of 4](https://github.com/vecxoz/vecstack/raw/master/pic/dia5.png "Step 2 of 4")
***
![Step 3 of 4](https://github.com/vecxoz/vecstack/raw/master/pic/dia6.png "Step 3 of 4")
***
![Step 4 of 4](https://github.com/vecxoz/vecstack/raw/master/pic/dia7.png "Step 4 of 4")

# Variant B. Animation

![Variant B. Animation](https://github.com/vecxoz/vecstack/raw/master/pic/animation2.gif "Variant B. Animation")
