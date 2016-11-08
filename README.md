# vecstack
Simple stacking package for Python and tutorial

***
So there is short stacking tutorial.
* Main idea of stacking: 
  *  Fit some model (1-st level) on initial train set.
  *  Predict initial train and test set with this model. 
  *  Make **new** train and test sets of this predictions.
  *  Fit some other model (2-nd level) on **new** train set and predict **new** test set.
  *  Prediction for **new** test set is our final target prediction.
* Main pitfall:
  *  If we will fit 1-st level model on train set and predict train set - we will get overfitting.
* Solution to overfitting:
  

***
![stack1](https://github.com/vecxoz/vecstack/blob/master/tutorial/dia1.png "Fold 1 of 3")
***
![stack2](https://github.com/vecxoz/vecstack/blob/master/tutorial/dia2.png "Fold 2 of 3")
***
![stack3](https://github.com/vecxoz/vecstack/blob/master/tutorial/dia3.png "Fold 3 of 3")
