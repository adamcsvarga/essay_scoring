# -*- coding: utf-8 -*-
"""
Created on Tue May 10 19:39:00 2016

Main training and testing module for automated essay scoring experiments

@author: Adam Varga
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
    # Read csv to pandas dataframe
    return pd.read_csv(filename, index_col='essay_id', low_memory=False)
    
def read_fulldata(filepath):
    # Read tsv to pandas dataframe
    return  pd.DataFrame.from_csv(filepath, sep="\t")
    
def train_and_xval(dataframe):
    """Training and cross-validation
    
    Input: pandas dataframe
    Output: quadratic weighted kappa score for results (floats)
    
    Uncomment certain blocks for different settings
    """
    
    ## Use all features    
    to_excl = ["essay_set", "essay", "domain1_score"]
    predictors = dataframe.columns.difference(to_excl)
    
    ## Use only dense features (w\ or w\out LSA)
    #predictors = ["essay_set", "essay", "domain1_score", "lemmacount", "modifier2np", "avglen", "mispell", "flesch", "smog", "fleschkinc", "aread", "dalechall", "diffwords", "linwrite", "gunning", "textstand"]
    #predictors = ["essay_set", "essay", "domain1_score", "lemmacount", "modifier2np", "avglen", "mispell", "flesch", "smog", "fleschkinc", "aread", "dalechall", "diffwords", "linwrite", "gunning", "textstand", "lsascore"]
    
    ## Use only sparse features    
    #to_excl = ["essay_set", "essay", "domain1_score", "lemmacount", "modifier2np", "avglen", "mispell", "flesch", "smog", "fleschkinc", "aread", "dalechall", "diffwords", "linwrite", "gunning", "textstand", "lsascore"]   
    #predictors = dataframe.columns.difference(to_excl)
    
    # plot feature importance
    selector = SelectKBest(f_classif, k=13)
    selector.fit(dataframe[predictors], dataframe["domain1_score"])
    scores = selector.pvalues_
    scores[scores == 0] = 1e-319
    scores = -np.log10(selector.pvalues_)
    
    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()
    plt.savefig('feas.png')
    plt.close()
    
    ## Recursive feature ranking
    rfe = RFE(estimator=alg, n_features_to_select=13, step=1)
    rfe.fit(dataframe[predictors], dataframe["domain1_score"])
    ranking = rfe.ranking_
    print(ranking, len(predictors), rfe.n_features_, rfe.support_, rfe.estimator_)
    
    # Choose learning algorithm    
    #alg = LinearRegression()
    #alg = SVR(kernel='linear')
    alg = GradientBoostingRegressor(min_samples_leaf=2)
    
    # Drop sparse features
    sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
    dn = pd.DataFrame(sel.fit_transform(dataframe[predictors]))

    # 10-fold cross-validation
    kf = KFold(dataframe.shape[0], n_folds=10, random_state=1)
    # train folds
    predictions = []
    for train, test in kf:
        train_predictors = (dn.iloc[train,:])
        train_target = dataframe["domain1_score"].iloc[train]
        alg.fit(train_predictors, train_target)
        test_predictions = alg.predict(dn.iloc[test,:])
        predictions.append(test_predictions)
    # concatenate fold
    predictions = np.concatenate(predictions, axis=0).round()   
   
    # Plot prediction vs. real value    
    fig, ax = plt.subplots()
    ax.scatter(dataframe["domain1_score"], predictions)
    ax.plot([dataframe["domain1_score"].min(), dataframe["domain1_score"].max()],[dataframe["domain1_score"].min(), dataframe["domain1_score"].max()], 'k--', lw=4)
    ax.set_xlabel('Expert score')
    ax.set_ylabel('Predicted score')    
    plt.show()
    fig.savefig('regr.png')
    plt.close(fig)
    
    # compute kappa    
    return qwk.quadratic_weighted_kappa(dataframe["domain1_score"], predictions)

def ia_agreement(dataframe):
    
    """Computes inter-annotator agreement
    Input: dataframe
    Output: qwk for inter-anotator agreement (float)"""
    return qwk.quadratic_weighted_kappa(dataframe["rater1_domain1"],\
    dataframe["rater2_domain1"])

if __name__ == '__main__':
    # demo on essay set 8 -- model vs. inter-annotatot agreement    
    df = read_data('8_full_lsa.csv')
    print(train_and_xval(df))
    
    ia = read_fulldata('data/training_set_rel3.tsv')
    print(ia_agreement(ia[ia.essay_set == 8]))
    
    
