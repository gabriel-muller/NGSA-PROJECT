# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:38:18 2020

@author: Benjamin Pommier
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import json
import pickle
import re

import sklearn
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split

def load_data(path, filename):
    with open(path + filename) as json_file:
        data = json.load(json_file)
    return data


def create_dataframe_unsorted(data, columns):
    array = np.zeros((len(data), len(columns)))
    for key, vals in data.items():
        for val in vals:
            i = int(key)
            j = columns.index(val)
            array[i,0] = i
            array[i,j] = 1 
    df = pd.DataFrame()
    for i,col in enumerate(columns):
        df[col] = array[:,i]
    df = df.set_index('Id')
    return df


def train(X, y, model, gridsearch, params, full_train, most_popular):

    #Keeping only the most important classes
    if most_popular is not None:
        idx = y.sum().sort_values(ascending=False).index[:most_popular]
        y = y.loc[:, idx]

    #Train test split
    if not full_train:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    else:
        X_train = X
        y_train = y
        
    #Gridsearch
    if gridsearch:
        params = params
        grid = GridSearchCV(estimator=model, param_grid=params, scoring='f1_weighted', n_jobs=-1, cv=5, verbose=50)
        grid.fit(X_train, y_train)
        # print('\n----- CV Results -----')
        # print(pd.DataFrame(grid.cv_results_))
        print('\n##### Optimal parameters #####')
        print(grid.best_params_)
        model = grid.best_estimator_
        
    #Training
    model.fit(X_train, y_train)
    print('-- TRAIN --')
    f1_train, roc_train = report(model, X_train, y_train)
    
    if not full_train:
        print('-- TEST --')
        f1_test, roc_test = report(model, X_test, y_test)    
    print('')
    
    return model, f1_train, f1_test


def report(model, X, y):
    
    #predictions
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)
    
    #F1 scores
    f1_macro = f1_score(y_true=y, y_pred=y_pred, average='macro')
    f1_micro = f1_score(y_true=y, y_pred=y_pred, average='micro')
    f1_weighted = f1_score(y_true=y, y_pred=y_pred, average='weighted')
    f1 = {'macro': f1_macro, 'micro': f1_micro, 'weighted': f1_weighted}
    print('F1 Scores: \n Macro : %.3f | Micro : %.3f | Weighted : %.3f'%(f1_macro, f1_micro, f1_weighted))
    
    #ROC
    roc_macro = roc_auc_score(y_true=y, y_score=y_score, average='macro')
    roc_micro = roc_auc_score(y_true=y, y_score=y_score, average='micro')
    roc_weighted = roc_auc_score(y_true=y, y_score=y_score, average='weighted')
    roc = {'macro': roc_macro, 'micro': roc_micro, 'weighted': roc_weighted}
    print('ROC Scores: \n Macro : %.3f | Micro : %.3f | Weighted : %.3f'%(roc_macro, roc_micro, roc_weighted))
    
    return f1, roc

        
def compare_embeddings(embeddings, y, model, gridsearch=False, params=None, full_train=False, most_popular=None):
    '''
    Compare the different embeddings by training a new model each time.

    Parameters
    ----------
    embeddings : dict
        Dict containing all the different embeddings with their respective values.
    y : array-like
        Ground truth label.

    Returns
    -------
     : results of the assessment
    '''
    
    i = 0
    for name, embed in embeddings.items():
        model, f1_train, f1_test = train(model=model, X=embed, y=y, gridsearch=gridsearch, 
                                params=params, full_train=full_train, most_popular=most_popular)
        if i == 0:
            results_train = pd.DataFrame(data=f1_train, index=[name])
            results_test = pd.DataFrame(data=f1_test, index=[name])
        else:
            temp = pd.DataFrame(data=f1_train, index=[name])
            results_train = pd.concat([results_train, temp], axis=0)
    
            temp = pd.DataFrame(data=f1_test, index=[name])
            results_test = pd.concat([results_test, temp], axis=0)
        i += 1
    
    print('#### TRAIN ####')
    print(results_train)
    print('#### TEST ####')
    print(results_test)
    
    return results_train, results_test

def clean_type(model):
    regex = re.compile("[^a-zA-Z]+")
    name = str(type(model)).split('.')[-1]
    name = regex.sub('', name)
    return name


#%%Main

### Loading data
path = '../data'
data_HR = load_data(path,'/HR_genres.json')
data_HU = load_data(path,'/HU_genres.json')
data_RO = load_data(path,'/RO_genres.json')

#get all genres
genres = []
for val in data_HR.values():
    genres += val
for val in data_HU.values():
    genres += val    
for val in data_RO.values():
    genres += val     
genres = list(set(genres))
genres.sort()
columns = ['Id'] + genres

#Country
country = ['HR','HU','RO']

### Loading GEMSEC features
embed_path = '../embeddings/'
X_HR_GEMSEC = pd.read_csv(embed_path+'HR_GEMSEC_embedding.csv')
X_HU_GEMSEC = pd.read_csv(embed_path+'HU_GEMSEC_embedding.csv')
X_RO_GEMSEC = pd.read_csv(embed_path+'RO_embedding_GEMSEC.csv')

### Loading GEMSEC With Regularization features
X_HR_GEMSECWithRegul = pd.read_csv(embed_path+'HR_GEMSECWithRegularization_embedding.csv')
X_HU_GEMSECWithRegul = pd.read_csv(embed_path+'HU_GEMSECWithRegularization_embedding.csv')
X_RO_GEMSECWithRegul = pd.read_csv(embed_path+'RO_GEMSECWithRegularization_embedding.csv')

### Loading DeepWalk features
X_HR_DeepWalk = pd.read_csv(embed_path+'HR_DeepWalk_embedding.csv')
X_HU_DeepWalk = pd.read_csv(embed_path+'HU_DeepWalk_embedding.csv')
X_RO_DeepWalk = pd.read_csv(embed_path+'RO_embedding_DW.csv')

### Loading DeepWalk With Regularization features
X_HR_DeepWalkWithRegularization = pd.read_csv(embed_path+'HR_DeepWalkWithRegularization_embedding.csv')
X_HU_DeepWalkWithRegularization = pd.read_csv(embed_path+'HU_DeepWalkWithRegularization_embedding.csv')
X_RO_DeepWalkWithRegularization = pd.read_csv(embed_path+'RO_embedding_DWR.csv')

### Loading Node2vec features
# X_HR_n2v = pd.read_csv(embed_path+'HR_node2vec_embedding.csv')
# X_HU_n2v = pd.read_csv(embed_path+'HU_node2vec_embedding.csv')
# X_RO_n2v = pd.read_csv(embed_path+'RO_node2vec_embedding.csv')

### Loading labels 
y_HR = create_dataframe_unsorted(data_HR, columns)
y_HU = create_dataframe_unsorted(data_HU, columns)
y_RO = create_dataframe_unsorted(data_RO, columns)
list_target = [y_HR, y_HU, y_RO]

### Initalizing the parameters for the models comparison

embeddings_HR = {'GEMSEC': X_HR_GEMSEC, 'GEMSEC With Regularization': X_HR_GEMSECWithRegul,
                 'DeepWalk': X_HR_DeepWalk, 'DeepWalk With Regularization': X_HR_DeepWalkWithRegularization} #, 'Node2Vec': X_HR_n2v}
embeddings_HU = {'GEMSEC': X_HU_GEMSEC, 'GEMSEC With Regularization': X_HU_GEMSECWithRegul,
                 'DeepWalk': X_HU_DeepWalk, 'DeepWalk With Regularization': X_HU_DeepWalkWithRegularization} #, 'Node2Vec': X_HU_n2v}
embeddings_RO = {'GEMSEC': X_RO_GEMSEC, 'GEMSEC With Regularization': X_RO_GEMSECWithRegul,
                 'DeepWalk': X_RO_DeepWalk, 'DeepWalk With Regularization': X_RO_DeepWalkWithRegularization} #, 'Node2Vec': X_RO_n2v}
list_embed = [embeddings_HR, embeddings_HU, embeddings_RO]

embed_y = list(zip(list_embed, list_target))

### Model used for the classification (4 combinaisons choisies)
weak_model_lg = LogisticRegression(n_jobs=-1)
weak_model_xgb = xgb.XGBClassifier()

meta_model_cc_lg = ClassifierChain(weak_model_lg)
meta_model_cc_xgb = ClassifierChain(weak_model_xgb)
meta_model_ovr_lg = OneVsRestClassifier(weak_model_lg, n_jobs=-1)
meta_model_ovr_xgb = OneVsRestClassifier(weak_model_xgb, n_jobs=-1)
list_models = [meta_model_cc_lg, meta_model_cc_xgb, meta_model_ovr_lg, meta_model_ovr_xgb]

# Parameters for the gridsearch
params_cc_lg = {'base_estimator__C': [0.1, 1, 10]}
params_cc_xbg = {'base_estimator__learning_rate': [0.1, 0.01]}
params_ovr_lg = {'estimator__C': [0.1, 1, 10]}
params_ovr_xgb = {'estimator__learning_rate': [0.1, 0.01]}
list_params = [params_cc_lg, params_cc_xbg, params_ovr_lg, params_ovr_xgb]

models_params = list(zip(list_models, list_params))

# Other important parameters
gridsearch = False
full_train = False
most_popular = 40

def run(list_data, model_params, gridsearch, full_train, most_popular, filename=None):
    results_train = dict()
    results_test = dict()
    i = 0
    for X, y in embed_y:
        cntry = country[i]
        i += 1
        for model, param in models_params:
            meta_model = clean_type(model)
            try:
                weak_model = clean_type(model.estimator)
            except AttributeError:
                weak_model = clean_type(model.base_estimator)
                
            print('##### Country: ' + cntry + ' | Meta model: ' + meta_model + ' | Weak model: ' + weak_model + ' #####')
            r_train, r_test = compare_embeddings(embeddings=X, y=y, model=model, 
                                     gridsearch=gridsearch, params=param, full_train=full_train,
                                     most_popular=most_popular)
            print('')

            results_train[(cntry, meta_model, weak_model)] = r_train
            results_test[(cntry, meta_model, weak_model)] = r_test
    
    #save the results
    pickle.dump(results_train, open('../outputs/train.pkl', 'wb'))
    pickle.dump(results_test, open('../outputs/test.pkl', 'wb'))
    
    return results_test, results_train

r_train, r_test = run(list_data=embed_y, model_params=models_params, gridsearch=gridsearch, 
    full_train=full_train, most_popular=most_popular)   
