# vecstack
Simple stacking package for Python and tutorial (look below).

***

# Stacking idea

1. We want to predict train and test sets with some 1-st level model(s), and then use this predictions as features for 2-nd level model.
2. Any model can be used as 1-st level model or 2-nd level model.
3. To avoid overfitting (for train set) we use cross-validation technique and in each fold we predict out-of-fold part of train set.
4. In each fold we predict full test set, so after complition of all folds we need to find mean of all test set predictions made in each fold.
5. Three pictures below describe one full cross-validation cycle (3 folds) for single 1-st level model. After its complition we get single train feature and single test feature to use with 2-nd level model.
6. We can repeat this cycle using other 1-st level models to get more features for 2-nd level model.

***

# Short stacking tutorial

## Main ideas
* Main idea of stacking: 
  * Fit some models (1-st level) on initial train set.
  * Predict initial train set and test set with this model. 
  * Make *new* train and test sets of this predictions.
  * Fit some other model (2-nd level) on *new* train set and predict *new* test set.
  * Prediction for *new* test set is our final target prediction.
* Main pitfall:
  * If we will fit 1-st level model on train set and predict train set - we will get overfitting.
* Solution to overfitting:
  * Use cross-validation technique to predict out-of-fold part of train set in each fold.
  
## Let's look on example below
* Here we have stacking implemented for single 1-st level model and 3-fold cross-validation.
* So there are 3 pictures - one for each fold.
* After complition of all three folds we will have one feature (one column) for *new* train set to fit 2-nd level model and one feature (one column) for *new* test set to predict with 2-nd level model.
* If we repeat this cycle with other 1-st level model - we will get another feature for 2-nd level model and so on.
* It's handy to watch this pictures successively in some image viewer, so you better see the changes from one picture to another.

***
![stack1](https://github.com/vecxoz/vecstack/blob/master/tutorial/dia1.png "Fold 1 of 3")
***
![stack2](https://github.com/vecxoz/vecstack/blob/master/tutorial/dia2.png "Fold 2 of 3")
***
![stack3](https://github.com/vecxoz/vecstack/blob/master/tutorial/dia3.png "Fold 3 of 3")
