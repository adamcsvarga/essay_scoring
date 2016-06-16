# -*- coding: utf-8 -*-
"""
Created on Tue May 10 19:39:00 2016

@author: vurga
"""

import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest, f_classif, RFE, \
VarianceThreshold
import quadratic_weighted_kappa as qwk
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor,\
 ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

def read_data(filename):
    
    return pd.read_csv(filename, index_col='essay_id', low_memory=False)
    
def read_fulldata(filepath):
    return  pd.DataFrame.from_csv(filepath, sep="\t")
    
def train_and_xval(dataframe):
    
    
    to_excl = ["essay_set", "essay", "domain1_score"]
    #to_excl = ["essay_set", "essay", "domain1_score", "lemmacount", "modifier2np", "avglen", "mispell", "flesch", "smog", "fleschkinc", "aread", "dalechall", "diffwords", "linwrite", "gunning", "textstand", "lsascore"]
    predictors = ["lemmacount", "modifier2np", "avglen", "mispell", "flesch", "smog", "fleschkinc", "aread", "dalechall", "diffwords", "linwrite", "gunning", "textstand"]
    predictors = dataframe.columns.difference(to_excl)
    
    
    # feature selection
#    selector = SelectKBest(f_classif, k=4)
#    selector.fit(dataframe[predictors], dataframe["domain1_score"])
#    scores = selector.pvalues_
#    scores[scores == 0] = 1e-319
#    scores = -np.log10(selector.pvalues_)
#    
#    plt.bar(range(len(predictors)), scores)
#    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
#    plt.show()
#    plt.savefig('feas.png')
#    plt.close()


    #predictors = ["avglen", "mispell", "modifier2np", "lemmacount", "dalechall"]
    
    #alg = LinearRegression()
    #alg = SVR(kernel='linear')
    alg = GradientBoostingRegressor(min_samples_leaf=2)
    
    # Drop sparse features
    #sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
    #dn = pd.DataFrame(sel.fit_transform(dataframe[predictors]))
    dn = dataframe[predictors]
    
    ## Feature ranking
#    rfe = RFE(estimator=alg, n_features_to_select=1, step=1)
#    rfe.fit(dataframe[predictors], dataframe["domain1_score"])
#    ranking = rfe.ranking_
#    print(ranking, len(predictors), rfe.n_features_, rfe.support_, rfe.estimator_)

    
    
    kf = KFold(dataframe.shape[0], n_folds=10, random_state=1)
    
    predictions = []
    for train, test in kf:
        train_predictors = (dn.iloc[train,:])
        train_target = dataframe["domain1_score"].iloc[train]
        alg.fit(train_predictors, train_target)
        test_predictions = alg.predict(dn.iloc[test,:])
        predictions.append(test_predictions)
        
        
    predictions = np.concatenate(predictions, axis=0).round()   
   
#    fig, ax = plt.subplots()
#    ax.scatter(dataframe["domain1_score"], predictions)
#    ax.plot([dataframe["domain1_score"].min(), dataframe["domain1_score"].max()],[dataframe["domain1_score"].min(), dataframe["domain1_score"].max()], 'k--', lw=4)
#    ax.set_xlabel('Expert score')
#    ax.set_ylabel('Predicted score')    
#    plt.show()
#    fig.savefig('regr.png')
#    plt.close(fig)
    
    return qwk.quadratic_weighted_kappa(dataframe["domain1_score"], predictions)

def ia_agreement(dataframe):
    
    return qwk.quadratic_weighted_kappa(dataframe["rater1_domain1"],\
    dataframe["rater2_domain1"])

if __name__ == '__main__':
    #df = read_data('8_full_lsa.csv')
    #print(train_and_xval(df))
    
    ia = read_fulldata('data/training_set_rel3.tsv')
    print(ia_agreement(ia[ia.essay_set == 8]))
    
    
