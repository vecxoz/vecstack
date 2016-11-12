# vecstack
Python package for stacking (machine learning technique).  
Can be used with any sklearn-like models.  
Can be used for classification and regression tasks.  
Below you can find [how to use](https://github.com/vecxoz/vecstack#how-to-use) module and explanation of [stacking concept](https://github.com/vecxoz/vecstack#stacking-concept) with pictures.  
Dependencies: *numpy*, *scipy*, *scikit-learn*.

# How to use

# Stacking concept

1. We want to predict train and test sets with some 1-st level model(s), and then use this predictions as features for 2-nd level model.  
2. Any model can be used as 1-st level model or 2-nd level model.
3. To avoid overfitting (for train set) we use cross-validation technique and in each fold we predict out-of-fold part of train set.
4. The common practice is to use from 3 to 10 folds.
5. In each fold we predict full test set, so after complition of all folds we need to find mean of all test set predictions made in each fold.
6. As an example we look at stacking implemented with single 1-st level model and 3-fold cross-validation.
7. Tree pictures below describe three folds of cross-validation. After complition of all three folds we get single train feature and single test feature to use with 2-nd level model.
8. We can repeat this cycle using other 1-st level models to get more features for 2-nd level model.
9. At the bottom you can see [GIF animation](https://github.com/vecxoz/vecstack/blob/master/README.md#animation).

***
![stack1](https://github.com/vecxoz/vecstack/blob/master/pic/dia1.png "Fold 1 of 3")
***
![stack2](https://github.com/vecxoz/vecstack/blob/master/pic/dia2.png "Fold 2 of 3")
***
![stack3](https://github.com/vecxoz/vecstack/blob/master/pic/dia3.png "Fold 3 of 3")
***

# Animation
![animation](https://github.com/vecxoz/vecstack/blob/master/pic/dia.gif "Animation")
