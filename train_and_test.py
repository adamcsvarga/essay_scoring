# -*- coding: utf-8 -*-
"""
Created on Tue May 10 19:39:00 2016

@author: vurga
"""

import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import quadratic_weighted_kappa as qwk

def read_data(filename):
    
    return pd.DataFrame.from_csv(filename)
    
def train_and_xval(dataframe):
    
    predictors = ["lemmacount", "modifier2np", "avglen", "mispell"]
    alg = LinearRegression()
    kf = KFold(dataframe.shape[0], n_folds=3, random_state=1)
    
    predictions = []
    for train, test in kf:
        train_predictors = (dataframe[predictors].iloc[train,:])
        train_target = dataframe["domain1_score"].iloc[train]
        alg.fit(train_predictors, train_target)
        
        test_predictions = alg.predict(dataframe[predictors].iloc[test,:])
        predictions.append(test_predictions)
        
    predictions = np.concatenate(predictions, axis=0).round()
    
    return qwk.quadratic_weighted_kappa(dataframe["domain1_score"], predictions)

if __name__ == '__main__':
    df = read_data('train_feas.csv')
    print(train_and_xval(df))
    
