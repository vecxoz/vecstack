# vecstack
Simple stacking package for Python and tutorial (look below).

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
* After 3 folds are completed we will have one feature (one column) for *new* train set to fit 2-nd level model and
  one feature (one column) for *new* test set to predict with 2-nd level model.
* If we repeat this cycle with other 1-st level model - we will get another feature for 2-nd level model and so on.

***
![stack1](https://github.com/vecxoz/vecstack/blob/master/tutorial/dia1.png "Fold 1 of 3")
***
![stack2](https://github.com/vecxoz/vecstack/blob/master/tutorial/dia2.png "Fold 2 of 3")
***
![stack3](https://github.com/vecxoz/vecstack/blob/master/tutorial/dia3.png "Fold 3 of 3")
