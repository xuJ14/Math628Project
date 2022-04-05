# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

#stacking
class StackingEnsmbleModels():
    def __init__(self, base_models, meta_model, n_folds=5):
        '''initial the models we use'''
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        
    def fit(self,X,y):
        X=np.array(X)
        y=np.array(y)
        self.base_models_= [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        # kfold = KFold(n_splits = self.n_folds, random_state=2022)
        kfold = KFold(n_splits = self.n_folds)

        level_1_prediction=np.zeros((X.shape[0],len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X,y):
                instance = model
                instance.fit(X[train_index],y[train_index])
                y_pred = instance.predict(X[holdout_index])
                self.base_models_[i].append(instance)
                level_1_prediction[holdout_index,i]=y_pred
                
        self.meta_model_.fit(level_1_prediction,y)
        return self
    
    def predict(self,X):
        meta_features=np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)
    

#Grid search to find the best parameters
def get_best_para_grid(estimators,X_train,y_train):
    best_para=dict()
    best_estimators = dict()
    for i in estimators.keys():
        if i=='rf':
            grid=GridSearchCV(estimators[i], param_grid={'n_estimators': [10,20,50,100,200,500], 'max_features': ['auto',0.2,0.4,0.6,0.8,1]}, cv=5)
            grid.fit(X_train, y_train)
            print("The best parameters for {} are {} with a score of {}".format(i,grid.best_params_, grid.best_score_))
            best_para['rf']=grid.best_params_
            best_estimators['rf']=grid.best_estimator_
        if i=='svc':
            grid = GridSearchCV(estimators[i], param_grid={'svc__kernel':['rbf','sigmod'], 'svc__C': [0.01, 0.1, 1, 10], 'svc__gamma': ['scale',0.01,0.1,0.5,1]}, cv=5)
            grid.fit(X_train, y_train)
            print("The best parameters for {} are {} with a score of {}".format(i,grid.best_params_, grid.best_score_))
            best_para['svc']=grid.best_params_
            best_estimators['svc']=grid.best_estimator_
        if i=='lgbm':
            grid=GridSearchCV(estimators[i], param_grid={'max_depth':[2,4,8],'learning_rate':[0.01,0.1,0.5,1,10],
                                                         'n_estimators':[20,50,100,200],'reg_lambda':[0,0.01,0.1,1,10]}, cv=5)
            grid.fit(X_train, y_train)
            print("The best parameters for {} are {} with a score of {}".format(i,grid.best_params_, grid.best_score_))
            best_para['lgbm']=grid.best_params_
            best_estimators['lgbm']=grid.best_estimator_
        if i=='etc':
            grid=GridSearchCV(estimators[i], param_grid={'max_depth':[None,2,4,8],'n_estimators':[20,50,100,200],
                                                        'max_features': ['auto',0.2,0.4,0.6,0.8,1]}, cv=5)
            grid.fit(X_train, y_train)
            print("The best parameters for {} are {} with a score of {}".format(i,grid.best_params_, grid.best_score_))
            best_para['etc']=grid.best_params_
            best_estimators['etc']=grid.best_estimator_
        if i=='xgb':
            grid=GridSearchCV(estimators[i], param_grid={'max_depth':[2,4,8],'learning_rate':[0.01,0.1,0.5,1,10],
                                                         'n_estimators':[20,50,100,200],'reg_lambda':[0.01,0.1,0.1,10]}, cv=5)
            grid.fit(X_train, y_train)
            print("The best parameters for {} are {} with a score of {}".format(i,grid.best_params_, grid.best_score_))
            best_para['xgb']=grid.best_params_
            best_estimators['xgb']=grid.best_estimator_
    return best_para,best_estimators

#Create model
def model():
    estimators = dict()
    estimators['rf'] = RandomForestClassifier()
    estimators['svc']=make_pipeline(StandardScaler(),SVC())
    estimators['lgbm']=LGBMClassifier()
    estimators['etc']=ExtraTreesClassifier()
    estimators['xgb']=XGBClassifier()
    estimators['clf']= StackingClassifier(
        estimators=list(estimators.items()), 
        final_estimator= LogisticRegression(),
        cv=5
        )
    return estimators

#score
def evaluate_model(model, X_train, y_train):
    cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
    score = cross_validate(model,X_train,y_train,scoring=['accuracy','f1','roc_auc','precision'],cv=cv)
    return score

#PCA
def pca(X_train,X_test,X_trace,n):
    model=PCA(n_components=n,whiten=True)
    x_train=model.fit_transform(X_train)
    x_test=model.transform(X_test)
    x_trace=model.transform(X_trace)
    component=model.components_
    return x_train,x_test,x_trace,component

def search_pca_n(X_train,X_test,y_train,y_test,estimators,final_estimator):
    accuracy=dict()
    for i in range(1,X_train.shape[1]+1):
        x_train,x_test,component=pca(X_train,X_test,i)
        model = StackingClassifier(estimators=list(estimators.items()),final_estimator=final_estimator,cv=5)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        accuracy[i]=metrics.accuracy_score(y_test,y_pred)
    return accuracy