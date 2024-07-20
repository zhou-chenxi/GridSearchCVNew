import re 
import pandas as pd 
import numpy as np
from sklearn.utils import all_estimators
import xgboost as xgb
import catboost 
import lightgbm 
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import ParameterGrid

# metrics 
from scipy.stats import ks_2samp
from sklearn.metrics import auc, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score


def evaluate_clf_helper(y_true, y_pred, sample_weight): 
    
    ks_score = ks_2samp(y_true, y_pred)
    auc_w = roc_auc_score(y_true, y_pred)
    precision_w = precision_score(y_true, y_pred, sample_weight = sample_weight)
    recall_w = recall_score(y_true, y_pred, sample_weight = sample_weight)
    f1_w = f1_score(y_true, y_pred, sample_weight = sample_weight)
    
    auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'Weighted Metrics': [np.nan, auc_w, precision_w, recall_w, f1_w], 
        'Unweighted Metrics': [ks_score.statistic, auc, precision, recall, f1]
    }


def evaluate_clf(estimator, X_train, y_train, X_test, y_test, class_weight): 
    
    if class_weight == 'balanced': 
        r = (1 - y_train).sum() / y_train.sum()
        sample_weight_test = np.ones_like(y_test) + (y_test == 1.0) * (r - 1)
        sample_weight_train = np.ones_like(y_train) + (y_train == 1.0) * (r - 1)
    else: 
        weight0 = class_weight[0]
        weight1 = class_weight[1]
        sample_weight_test = np.ones_like(y_test) + (y_test == 1.0) * (weight1 - 1)
        sample_weight_train = np.ones_like(y_train) + (y_train == 1.0) * (weight1 - 1)
    
    # test 
    y_test_pred = estimator.predict(X_test)
    test_results = evaluate_clf_helper(y_test, y_test_pred, sample_weight_test)
    
    # train
    y_train_pred = estimator.predict(X_train)
    train_results = evaluate_clf_helper(y_train, y_train_pred, sample_weight_train)
    
    metrics = pd.DataFrame(
        {'Weighted Metrics (Test)': test_results['Weighted Metrics'], 
         'Unweighted Metrics (Test)': test_results['Unweighted Metrics'], 
         'Weighted Metrics (Train)': train_results['Weighted Metrics'], 
         'Unweighted Metrics (Train)': train_results['Unweighted Metrics']
        }, 
        index = ['KS Score', 'AUC', 'Precision', 'Recall', 'F1']
    )
    
    return metrics


class GridSearchCV_NEW(): 
    
    def __init__(self, estimator, cv_folds, param_grid = None, param_grid_list = None, class_weight_xgboost = None): 

        self.estimator = estimator
        
        if param_grid and not isinstance(param_grid, dict): 
            raise TypeError(f'param_grid should be a dict, but got a {param_grid.__class__.__name__}.')
        
        self.param_grid = param_grid

        if not param_grid and not param_grid_list: 
            raise ValueError('param_grid and param_grid_list are both None. Please provide values for exactly one of them. ')
        elif param_grid and param_grid_list: 
            raise ValueError('param_grid and param_grid_list are both provided. Please provide values for exactly one of them. ')
        
        if not param_grid_list: 
            if estimator in [i[1] for i in all_estimators()]: 
                self.param_grid_list = ParameterGrid(param_grid)
            elif estimator == xgb.XGBClassifier: 
                self.param_grid_list = ParameterGrid({k: v for (k, v) in param_grid.items() if k != 'class_weight'})
            elif estimator == catboost.CatBoostClassifier: 
                self.param_grid_list = ParameterGrid(param_grid)
            elif estimator == lightgbm.LGBMClassifier: 
                self.param_grid_list = ParameterGrid(param_grid)
            else: 
                raise ValueError('estimator provided is not in the list of desired estimators.')
        else: 

            # check whether the keys of each element in param_grid_list are identical 
            if len(set(map(len, param_grid_list))) != 1: 
                raise ValueError('Each item in param_grid_list must have the same parameter keys.')
            if len(set([j for i in param_grid_list for j in list(i.keys())])) != len(param_grid_list[0]): 
                raise ValueError('Each item in param_grod_list must have the same parameter keys.')
            self.param_grid_list = param_grid_list

        if self.estimator == xgb.XGBClassifier: 
            if param_grid and 'class_weight' in param_grid: 
                if class_weight_xgboost: 
                    raise ValueError('Please remove class_weight from param_grid and supply it in class_weight_xgboost.')
                else:
                    class_weight_xgboost = param_grid['class_weight']
                    del param_grid['class_weight']
            if param_grid_list and 'class_weight' in set([j for i in param_grid_list for j in list(i.keys())]): 
                if class_weight_xgboost: 
                    raise ValueError('Please remove class_weight from param_grid and supply it in class_weight_xgboost.')
        
        class_weight_flag1 = 'class_weight' in param_grid.keys() if param_grid else False 
        class_weight_flag2 = 'class_weight' in set([j for i in param_grid_list for j in list(i.keys())]) if param_grid_list else False 
        class_weight_flag3 = True if (self.estimator == xgb.XGBClassifier and class_weight_xgboost) else False 
        self.class_weight_flag = (class_weight_flag1 or class_weight_flag2 or class_weight_flag3)
        
        if self.estimator == xgb.XGBClassifier and class_weight_xgboost: 
             self.class_weight_xgboost = class_weight_xgboost
        
        if self.estimator != xgb.XGBClassifier: 
            print(f'A total of {len(self.param_grid_list)} model configurations will be tried.')
        else: 
            print(f'A total of {len(self.param_grid_list) * len(self.class_weight_xgboost)} model configurations will be tried.')
        
        if not isinstance(cv_folds, int): 
            raise TypeError(f'cv_folds must be an integer, but got a {cv_folds.__class__.__name__}.')
        self.cv_folds = cv_folds
    
    def _stratified_sampling(self, X, y, random_state = 0): 
        
        X = X.reset_index(drop = True)
        y = pd.DataFrame(y, columns = ['y'])
        df = pd.concat([X, y], axis = 1)
        
        if len(y['y'].unique()) != 2: 
            raise ValueError(f'The current implementation only works 2 classes, but got {len(df.y.unique())} classes.')

        # stratified sampling - assume there are only 2 class labels 
        df0 = df[df.y == 0].reset_index(drop = True)
        df1 = df[df.y == 1].reset_index(drop = True)
        
        np.random.seed(random_state)
        new_df0 = pd.concat(
            [df0, 
             pd.Series(
                 np.random.choice(list(range(self.cv_folds)), size = df0.shape[0]), 
                 index = df0.index, 
                 name = 'cv_label', 
                 dtype = 'category'
             )], axis = 1)
        
        new_df1 = pd.concat(
            [df1, 
             pd.Series(
                 np.random.choice(list(range(self.cv_folds)), size = df1.shape[0]), 
                 index = df1.index, 
                 name = 'cv_label', 
                 dtype = 'category'
             )], axis = 1)
        
        new_df = pd.concat([new_df1, new_df0])
        # print(new_df.groupby('cv_label').size())
        
        return new_df
    
    
    def fit(self, X, y, metric, enable_categorical = True, random_state = 0): 
        
        if self.estimator in [i[1] for i in all_estimators()]: 
            return self.fit_sklearn(X, y, metric, random_state)
        elif self.estimator == xgb.XGBClassifier: 
            return self.fit_xgboost(X, y, metric, enable_categorical, random_state)
        elif self.estimator == catboost.CatBoostClassifier: 
            return self.fit_catboost(X, y, metric, random_state)
        elif self.estimator == lightgbm.LGBMClassifier: 
            return self.fit_lightgbm(X, y, metric, random_state)
        # probably create an attribute to specify which kind of clf 
        
    
    def fit_sklearn(self, X, y, metric, random_state = 0): 
        
        new_df = self._stratified_sampling(X, y, random_state)
        
        # cross validation 
        cv_result = {}
    
        for param in self.param_grid_list: 

            print('=' * 80)
            print(f'Current parameter: {param}')
            
            if self.class_weight_flag: 
                weighted_score = []
                
            unweighted_score = []
            
            for fold in sorted(new_df.cv_label.unique()): 
                
                print('-' * 80)
                print(f'Fold number = {fold}')
                
                train_subdf = new_df[new_df.cv_label != fold]
                test_subdf = new_df[new_df.cv_label == fold]

                clf = self.estimator(random_state = random_state, **param)
                
                # fit the model 
                clf.fit(
                    X = train_subdf[[i for i in train_subdf.columns if i not in ['y', 'cv_label']]], 
                    y = train_subdf['y']
                )
                y_test_pred = clf.predict(test_subdf[[i for i in test_subdf.columns if i not in ['y', 'cv_label']]])
                
                # evaluate stage 
                if self.class_weight_flag: 
                    if param['class_weight'] != 'balanced': 
                        sample_weight = (np.ones_like(test_subdf.y) + 
                                         (test_subdf.y.to_numpy() == 1.0) * (param['class_weight'][1] - 1))
                    elif param['class_weight'] == 'balanced':
                        sample_weight = (np.ones_like(test_subdf.y) + 
                                         (test_subdf.y.to_numpy() == 1.0) * ((1 - y).sum() / y.sum() - 1))

                    weighted_metric = metric(test_subdf.y, y_test_pred, sample_weight = sample_weight)
                    weighted_score.append(weighted_metric)

                unweighted_metric = metric(test_subdf.y, y_test_pred)
                unweighted_score.append(unweighted_metric)

            cv_result_key = '|'.join([k + '=' + str(v) for (k, v) in param.items()])
            if self.class_weight_flag: 
                cv_result[cv_result_key] = {
                    'weighted_metric': weighted_score, 
                    'weighted_metric_avg': np.array(weighted_score).mean(), 
                    'unweighted_metric': unweighted_score, 
                    'unweighted_metric_avg': np.array(unweighted_score).mean()
                }
            else: 
                cv_result[cv_result_key] = {
                    'unweighted_metric': unweighted_score, 
                    'unweighted_metric_avg': np.array(unweighted_score).mean()
                }
        
        # the best combination of hyper-parameter 
        # assessed by unweighted metric
        unweighted_r = [v['unweighted_metric_avg'] for (k, v) in cv_result.items()]
        unweighted_best_param = self.param_grid_list[np.argmax(unweighted_r)]
        # run the best model 
        clf_unweighted = self.estimator(random_state = random_state, **unweighted_best_param)
        clf_unweighted.fit(X = X, y = y)
        self.best_unweighted_estimator = clf_unweighted

        # assessed by weighted metric 
        if self.class_weight_flag: 
            weighted_r = [v['weighted_metric_avg'] for (k, v) in cv_result.items()]
            weighted_best_param = self.param_grid_list[np.argmax(weighted_r)]
            
            # run the best model 
            clf_weighted = self.estimator(random_state = random_state, **weighted_best_param)
            clf_weighted.fit(X = X, y = y)
            self.best_weighted_estimator = clf_weighted

            return {
                'cv_result': cv_result, 
                'best_model_weighted': clf_weighted, 
                'best_model_unweighted': clf_unweighted
            }
        else: 
            return {
                'cv_result': cv_result, 
                'best_model_unweighted': clf_unweighted
            }
    
    def _compute_sample_weight(self, y, weight): 
        
        # weight should be a dict or 'balanced'
        if not isinstance(weight, dict): 
            raise ValueError('weight must be a dict.')
        
        if len(weight) != 2: 
            raise ValueError('weight should contain exactly 2 components.')
        
        y = pd.Series(y)
        
        return np.ones_like(y) + (y.to_numpy() == 1.0) * (weight[1] / weight[0] - 1)

    
    def fit_catboost(self, X, y, metric, random_state = 0): 
        
        new_df = self._stratified_sampling(X, y, random_state)
        
        # cross validation 
        cv_result = {}
    
        for param in self.param_grid_list: 

            print('=' * 80)
            print(f'Current parameter: {param}')
            
            if self.class_weight_flag: 
                weighted_score = []
                
            unweighted_score = []
            
            for fold in sorted(new_df.cv_label.unique()): 
                
                print('-' * 80)
                print(f'Fold number = {fold}')
                
                train_subdf = new_df[new_df.cv_label != fold]
                test_subdf = new_df[new_df.cv_label == fold]
                
                if 'class_weights' in param.keys() and param['class_weights'] == 'balanced': 
                    param['class_weights'] = {0: 1, 1: (1 - y).sum() / y.sum()}

                clf = self.estimator(random_state = random_state, **param)
                
                # fit the model 
                clf.fit(
                    X = train_subdf[[i for i in train_subdf.columns if i not in ['y', 'cv_label']]], 
                    y = train_subdf['y'], 
                    cat_features = [i for i in train_subdf.columns if i not in ['y', 'cv_label'] and train_subdf[i].dtype.name == 'category']
                )
                y_test_pred = clf.predict(
                    test_subdf[[i for i in test_subdf.columns if i not in ['y', 'cv_label']]]
                )
                
                # evaluate stage 
                if self.class_weight_flag: 
                    if param['class_weights'] != 'balanced': 
                        sample_weight = (np.ones_like(test_subdf.y) + 
                                         (test_subdf.y.to_numpy() == 1.0) * (param['class_weights'][1] - 1))
                    elif param['class_weights'] == 'balanced':
                        sample_weight = (np.ones_like(test_subdf.y) + 
                                         (test_subdf.y.to_numpy() == 1.0) * ((1 - y).sum() / y.sum() - 1))

                    weighted_metric = metric(test_subdf.y, y_test_pred, sample_weight = sample_weight)
                    weighted_score.append(weighted_metric)

                unweighted_metric = metric(test_subdf.y, y_test_pred)
                unweighted_score.append(unweighted_metric)

            cv_result_key = '|'.join([k + '=' + str(v) for (k, v) in param.items()])
            if self.class_weight_flag: 
                cv_result[cv_result_key] = {
                    'weighted_metric': weighted_score, 
                    'weighted_metric_avg': np.array(weighted_score).mean(), 
                    'unweighted_metric': unweighted_score, 
                    'unweighted_metric_avg': np.array(unweighted_score).mean()
                }
            else: 
                cv_result[cv_result_key] = {
                    'unweighted_metric': unweighted_score, 
                    'unweighted_metric_avg': np.array(unweighted_score).mean()
                }
        
        # the best combination of hyper-parameter 
        # assessed by unweighted metric
        unweighted_r = [v['unweighted_metric_avg'] for (k, v) in cv_result.items()]
        unweighted_best_param = self.param_grid_list[np.argmax(unweighted_r)]
        self.best_model_param_unweighted = unweighted_best_param
        if 'class_weights' in unweighted_best_param.keys() and unweighted_best_param['class_weights'] == 'balanced': 
            unweighted_best_param['class_weights'] = {0: 1, 1: (1 - y).sum() / y.sum()}
            
        # run the best model 
        clf_unweighted = self.estimator(random_state = random_state, **unweighted_best_param)
        clf_unweighted.fit(
            X = X,
            y = y, 
            cat_features = [i for i in X.columns if X[i].dtype.name == 'category']
        )
        self.best_unweighted_estimator = clf_unweighted

        # assessed by weighted metric 
        if self.class_weight_flag: 
            weighted_r = [v['weighted_metric_avg'] for (k, v) in cv_result.items()]
            weighted_best_param = self.param_grid_list[np.argmax(weighted_r)]
            self.best_model_param_weighted = weighted_best_param
            if 'class_weights' in weighted_best_param.keys() and weighted_best_param['class_weights'] == 'balanced': 
                weighted_best_param['class_weights'] = {0: 1, 1: (1 - y).sum() / y.sum()}
                
            # run the best model 
            clf_weighted = self.estimator(random_state = random_state, **weighted_best_param)
            clf_weighted.fit(
                X = X, 
                y = y, 
                cat_features = [i for i in X.columns if X[i].dtype.name == 'category']
            )
            self.best_weighted_estimator = clf_weighted

            return {
                'cv_result': cv_result, 
                'best_model_weighted': clf_weighted, 
                'best_model_unweighted': clf_unweighted
            }
        else: 
            return {
                'cv_result': cv_result, 
                'best_model_unweighted': clf_unweighted
            }

    
    def fit_lightgbm(self, X, y, metric, random_state = 0): 
        
        return self.fit_sklearn(X, y, metric, random_state)


    def fit_xgboost(self, X, y, metric, enable_categorical = True, random_state = 0): 
        
        if self.class_weight_flag: 
            return self.fit_xgboost_weighted(X, y, metric, enable_categorical = True, random_state = 0)
        else: 
            return self.fit_xgboost_weighted(X, y, metric, enable_categorical = True, random_state = 0)
        
    def fit_xgboost_unweighted(self, X, y, metric, enable_categorical = True, random_state = 0): 
        
        new_df = self._stratified_sampling(X, y)
        
        # cross validation 
        cv_result = {}
        
        for param in self.param_grid_list: 
            
            print('=' * 80)
            print(f'Current parameter: {param}')
            
            for fold in sorted(new_df.cv_label.unique()): 
                
                print('-' * 80)
                print(f'Fold number = {fold}')
                
                train_subdf = new_df[new_df.cv_label != fold]
                test_subdf = new_df[new_df.cv_label == fold]
                
                clf = self.estimator(random_state = random_state, **param)
                
                dtrain = xgb.DMatrix(
                    train_subdf[[i for i in train_subdf.columns if i not in ['y', 'cv_label']]], 
                    label = train_subdf['y'], 
                    enable_categorical = enable_categorical
                )
                dtest = xgb.DMatrix(
                    test_subdf[[i for i in test_subdf.columns if i not in ['y', 'cv_label']]], 
                    enable_categorical = enable_categorical
                )
                print(f'test whether dtrain and dtest have the same columns: {dtrain.get_data().shape == dtest.get_data().shape}')
                
                clf.fit(
                    X = dtrain.get_data(), 
                    y = dtrain.get_label()
                )
                y_test_pred = clf.predict(dtest.get_data())
                
                unweighted_metric = metric(test_subdf.y, y_test_pred)
                unweighted_score.append(unweighted_metric)
                
            cv_result_key = '|'.join([k + '=' + str(v) for (k, v) in param.items()])
            cv_result[cv_result_key] = {
                'unweighted_metric': unweighted_score, 
                'unweighted_metric_avg': np.array(unweighted_score).mean()
            }
        
        # run the best model 
        unweighted_best_param_temp = [i.split('=') for i in unweighted_best_param.split('|')]
        unweighted_best_param_fun = {item[0]: item[1] for item in unweighted_best_param_temp}
        for (key, value) in unweighted_best_param_fun.items(): 
            if value[0].isdigit(): 
                if '.' in value: 
                    unweighted_best_param_fun[key] = float(value)
                else: 
                    unweighted_best_param_fun[key] = int(value)
            else: 
                unweighted_best_param_fun[key] = value
                
        clf_unweighted = self.estimator(random_state = random_state, **unweighted_best_param_fun)    
        self.best_unweighted_estimator = clf_unweighted
        
        return {
            'cv_result': cv_result, 
            'best_model_param_unweighted': unweighted_best_param, 
            'best_model_unweighted': clf_unweighted
            }
    
    def fit_xgboost_weighted(self, X, y, metric, enable_categorical = True, random_state = 0): 
        
        new_df = self._stratified_sampling(X, y)
        
        # cross validation 
        cv_result = {}
        
        for param in self.param_grid_list: 
            
            print('=' * 80)
            print(f'Current parameter: {param}')

            for w in self.class_weight_xgboost:
                
                unweighted_score = []
                weighted_score = []
                print(f'weight={w}')
                
                for fold in sorted(new_df.cv_label.unique()): 
                
                    print('-' * 80)
                    print(f'Fold number = {fold}')
                    
                    train_subdf = new_df[new_df.cv_label != fold]
                    test_subdf = new_df[new_df.cv_label == fold]
                    
                    clf = self.estimator(random_state = random_state, **param)
                    
                    dtrain = xgb.DMatrix(
                        train_subdf[[i for i in train_subdf.columns if i not in ['y', 'cv_label']]], 
                        label = train_subdf['y'], 
                        enable_categorical = enable_categorical
                    )
                    dtest = xgb.DMatrix(
                        test_subdf[[i for i in test_subdf.columns if i not in ['y', 'cv_label']]], 
                        enable_categorical = enable_categorical
                    )
                    # print(f'test whether dtrain and dtest have the same columns: {dtrain.get_data().shape, dtest.get_data().shape}')
                    
                    if w != 'balanced': 
                        
                        sample_weight_test = self._compute_sample_weight(y = test_subdf.y, weight = w)
                        sample_weight_train = self._compute_sample_weight(y = train_subdf.y, weight = w)
                        
                    elif w == 'balanced':
                        
                        ratio = (1. - y).sum() / y.sum()
                        sample_weight_test = self._compute_sample_weight(
                            y = test_subdf.y, 
                            weight = {0: 1, 1: ratio}
                        )
                        sample_weight_train = self._compute_sample_weight(
                            y = train_subdf.y, 
                            weight = {0: 1, 1: ratio}
                        )
                        
                    # print(f'sample_weight_train={sample_weight_train}')
                    # print(dtrain.get_label())
                    # print(f'sample_weight_test={sample_weight_test}')
                    # print(dtest.get_label())
                    
                    clf.fit(
                        X = dtrain.get_data(), 
                        y = dtrain.get_label(), 
                        sample_weight = sample_weight_train
                    )
                    y_test_pred = clf.predict(dtest.get_data())
                    
                    weighted_metric = metric(
                        test_subdf.y, 
                        y_test_pred, 
                        sample_weight = sample_weight_test)
                    weighted_score.append(weighted_metric)
                    
                    unweighted_metric = metric(test_subdf.y, y_test_pred)
                    unweighted_score.append(unweighted_metric)
                        
                cv_result_key = '|'.join([k + '=' + str(v) for (k, v) in param.items()]) + f'|class_weight={w}'
                cv_result[cv_result_key] = {
                    'weighted_metric': weighted_score, 
                    'weighted_metric_avg': np.array(weighted_score).mean(), 
                    'unweighted_metric': unweighted_score, 
                    'unweighted_metric_avg': np.array(unweighted_score).mean()
                }
                print("'" + cv_result_key + "': ")
                print(cv_result[cv_result_key])
                
        # the best combination of hyper-parameter 
        # assessed by unweighted metric
        unweighted_r = [v['unweighted_metric_avg'] for (k, v) in cv_result.items()]
        unweighted_best_param = list(cv_result.keys())[np.argmax(unweighted_r)]
        self.best_model_param_unweighted = unweighted_best_param
        
        ddata = xgb.DMatrix(X, y, enable_categorical = enable_categorical)
        
        # run the best model 
        if bool(re.search('class_weight', unweighted_best_param)): 
            
            unweighted_best_param0 = unweighted_best_param.split('class_weight=')[0]
            unweighted_best_param1 = unweighted_best_param.split('class_weight=')[1]

            if unweighted_best_param0.endswith('|'): 
                unweighted_best_param0 = unweighted_best_param0[:-1]

            unweighted_best_param0_temp = [i.split('=') for i in unweighted_best_param0.split('|')]

            print(unweighted_best_param0_temp)
            unweighted_best_param0_fun = {item[0]: item[1] for item in unweighted_best_param0_temp}
            for (key, value) in unweighted_best_param0_fun.items(): 
                if value[0].isdigit(): 
                    if '.' in value: 
                        unweighted_best_param0_fun[key] = float(value)
                    else: 
                        unweighted_best_param0_fun[key] = int(value)
                else: 
                    unweighted_best_param0_fun[key] = value
            
            clf_unweighted = self.estimator(random_state = random_state, **unweighted_best_param0_fun)
            if unweighted_best_param1 == 'balanced': 
                ratio = (1 - y).sum() / y.sum()
                sample_weight = self._compute_sample_weight(y = y, weight = {0: 1, 1: ratio})
                clf_unweighted.fit(ddata.get_data(), ddata.get_label(), sample_weight = sample_weight)
            else: 
                sample_weight = self._compute_sample_weight(y = y, weight = eval(unweighted_best_param1))
                clf_unweighted.fit(ddata.get_data(), ddata.get_label(), sample_weight = sample_weight)
                
        else: 
            
            unweighted_best_param_temp = [i.split('=') for i in unweighted_best_param.split('|')]
            unweighted_best_param_fun = {item[0]: item[1] for item in unweighted_best_param_temp}
            for (key, value) in unweighted_best_param_fun.items(): 
                if value[0].isdigit(): 
                    if '.' in value: 
                        unweighted_best_param_fun[key] = float(value)
                    else: 
                        unweighted_best_param_fun[key] = int(value)
                else: 
                    unweighted_best_param_fun[key] = value
                    
            clf_unweighted = self.estimator(random_state = random_state, **unweighted_best_param_fun)
            
        self.best_unweighted_estimator = clf_unweighted
        
        # assessed by weighted metric 
        if self.class_weight_flag: 
            weighted_r = [v['weighted_metric_avg'] for (k, v) in cv_result.items()]
            weighted_best_param = list(cv_result.keys())[np.argmax(weighted_r)]
            
            self.best_model_param_weighted = weighted_best_param
            
            weighted_best_param0 = weighted_best_param.split('class_weight=')[0]
            weighted_best_param1 = weighted_best_param.split('class_weight=')[1]

            if weighted_best_param0.endswith('|'): 
                weighted_best_param0 = weighted_best_param0[:-1]
            
            weighted_best_param0_temp = [i.split('=') for i in weighted_best_param0.split('|')]
            weighted_best_param0_fun = {item[0]: item[1] for item in weighted_best_param0_temp}
            for (key, value) in weighted_best_param0_fun.items(): 
                if value[0].isdigit(): 
                    if '.' in value: 
                        weighted_best_param0_fun[key] = float(value)
                    else: 
                        weighted_best_param0_fun[key] = int(value)
                else: 
                    weighted_best_param0_fun[key] = value
            
            clf_weighted = self.estimator(random_state = random_state, **weighted_best_param0_fun)
            if weighted_best_param1 == 'balanced': 
                ratio = (1 - y).sum() / y.sum()
                sample_weight = self._compute_sample_weight(y = y, weight = {0: 1, 1: ratio})
                clf_weighted.fit(ddata.get_data(), ddata.get_label(), sample_weight = sample_weight)
            else: 
                print(weighted_best_param1)
                sample_weight = self._compute_sample_weight(y = y, weight = eval(weighted_best_param1))
                clf_weighted.fit(ddata.get_data(), ddata.get_label(), sample_weight = sample_weight)
                
            self.best_weighted_estimator = clf_weighted
            
            return {
                'cv_result': cv_result, 
                'best_model_param_weighted': weighted_best_param, 
                'best_model_param_unweighted': unweighted_best_param, 
                'best_model_weighted': clf_weighted, 
                'best_model_unweighted': clf_unweighted
            }
        else: 
            return {
                'cv_result': cv_result, 
                'best_model_param_unweighted': unweighted_best_param, 
                'best_model_unweighted': clf_unweighted
            }
    
    def predict(self, newX): 
        
        if self.estimator in [i[1] for i in all_estimators()]: 
            y_pred_unweighted = self.best_unweighted_estimator.predict(newX)
            if self.class_weight_flag: 
                y_pred_weighted = self.best_weighted_estimator.predict(newX)
                return {'y_pred_unweighted': y_pred_unweighted, 'y_pred_weighted': y_pred_weighted}
            return y_pred_unweighted
        
        elif self.estimator == xgb.XGBClassifier: 
            
            d_newX = xgb.DMatrix(newX, enable_categorical = True)
            y_pred_unweighted = self.best_unweighted_estimator.predict(newX)
            if self.class_weight_flag: 
                y_pred_weighted = self.best_weighted_estimator.predict(newX)
                return {'y_pred_unweighted': y_pred_unweighted, 'y_pred_weighted': y_pred_weighted}
            return y_pred_unweighted
        
        elif self.estimator == catboost.CatBoostClassifier: 
            
            y_pred_unweighted = self.best_unweighted_estimator.predict(newX)
            if self.class_weight_flag: 
                y_pred_weighted = self.best_weighted_estimator.predict(newX)
                return {'y_pred_unweighted': y_pred_unweighted, 'y_pred_weighted': y_pred_weighted}
            return y_pred_unweighted
        
        elif self.estimator == lightgbm.LGBMClassifier: 
            
            y_pred_unweighted = self.best_unweighted_estimator.predict(newX)
            if self.class_weight_flag: 
                y_pred_weighted = self.best_weighted_estimator.predict(newX)
                return {'y_pred_unweighted': y_pred_unweighted, 'y_pred_weighted': y_pred_weighted}
            return y_pred_unweighted
          
