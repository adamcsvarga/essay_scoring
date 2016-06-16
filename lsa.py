# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 18:01:20 2016

Module for LSA-based scoring

@author: Adam Varga
"""

from __future__ import print_function
import sklearn
import numpy as np
# Import all of the scikit learn stuff
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
import pandas as pd
import warnings
# Suppress warnings from pandas library
warnings.filterwarnings("ignore", category=DeprecationWarning,
module="pandas", lineno=570)

def compute(df):
    
    """LSA computation
    
    Input: pandas dataframe
    Output: score of the most similar essay to the first essay
    """
    
    # get file's index
    ind = df.name  
    es = df[0]
    essay = df[1]
    
    trainfiles = \
    pd.DataFrame.from_csv('data/training_set_rel3.tsv',\
    sep="\t")
    # drop test file from training if present
    trainfiles.drop(ind, inplace=True)
    
    scores = trainfiles[trainfiles.essay_set == es].domain1_score
    # create training matrix
    testfile = pd.Series(essay)    
    trainfiles = trainfiles[trainfiles.essay_set == es].essay
    data = testfile.append(trainfiles)
    
    
    # create DTM
    vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
    dtm = vectorizer.fit_transform(data)
        
    
    # Fit LSA
    lsa = TruncatedSVD(100, algorithm = 'randomized')
    dtm_lsa = lsa.fit_transform(dtm)
    dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
    
    similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)
    simdf = pd.DataFrame(similarity,index=data.index, columns=data.index)
    
    closest_id = simdf.iloc[0][1:].idxmax()
    
    return scores[closest_id]
    
        
if __name__ == '__main__':
    
    # example computation
    compute("computers are cool.")
    
    
    
    
    
    
