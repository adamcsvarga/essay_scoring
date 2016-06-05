# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 18:01:20 2016

Module for LSA-based scoring

@author: Adam Varga
"""

from __future__ import print_function
import sklearn
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

def compute(essay_id, es, essay):
    
    trainfiles = \
    pd.DataFrame.from_csv('data/training_set_rel3.tsv',\
    sep="\t")
    
    trainfiles = trainfiles[trainfiles.essay_set == es].essay
    
    # create DTM
    vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
    dtm = vectorizer.fit_transform(trainfiles)
    
    # Fit LSA
    lsa = TruncatedSVD(2, algorithm = 'randomized')
    dtm_lsa = lsa.fit_transform(dtm)
    dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
    
    
    
if __name__ == '__main__':
    compute(1,1,"dfdfsd")
    
    
    
    
    
    
    
    
