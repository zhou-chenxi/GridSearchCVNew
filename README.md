# GridSearchCVNew
A new module to apply grid search for tuning hyperparameters in machine learning algorithms 

When we fit a classifier, it is a common practice to supply the `class_weight` parameter, which reflects our belief that misclassifying an observation in the positive class as negative does not have the same penalty as the reverse case. Typically, `estimator`s in `scikit-learn` allow users to specify `class_weight` as a parameter. When this happens, the classifier should be evaluated using the corresponding `class_weight` as well. However, the current [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) function in `scikit-learn`, a popular tool for evaluating models of various configurations (combinations of hyperparameters) and selecting the best possible model, fails to take the `class_weight` into consideration. In order to remedy this deficiency, the [`GridSearchCVNew`](https://github.com/zhou-chenxi/GridSearchNew/blob/main/GridSearchCVNew/GridSearchCVNew.py) module has been written and consider `class_weight` when the model is both fitted and evaluated. 

Currently, [`GridSearchCVNew`](https://github.com/zhou-chenxi/GridSearchNew/blob/main/GridSearchCVNew/GridSearchCVNew.py) only works for binary classification problems, but will be extended to multi-class classification problems. It supports all classifiers in `scikit-learn`, as well as [`xgb.XGBClassifier`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier), [catboost.CatBoostClassifier](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier), and [lightgbm.LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html). 

## Meta

Chenxi Zhou – chenxizhou.jayden@gmail.com

[https://github.com/zhoucx-chenxi/GridSearchCVNew](https://github.com/zhou-chenxi/GridSearchCVNew)
