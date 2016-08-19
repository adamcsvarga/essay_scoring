# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:59:25 2016

Feature extraction module for automated essay scoring

v0.0 -- baseline (May 13 2016)
v0.1 -- added readability score features (June 2 2016)
v0.2 -- added PoS ngrams (June 3 2016)
v0.3 -- added LSA scores (June 4 2016)

@author: Adam Varga
"""

from textstat.textstat import textstat
import pandas as pd
import lemma_count as lc, lsa
import modifier_count as mc, spellcheck as sp
from sklearn.feature_extraction.text import CountVectorizer

def read_data(filepath):
    """Read CSV to dataframe
    
    Input: path of the tab-separated CSV (str)
    Output: pandas dataframe"""
    return  pd.DataFrame.from_csv(filepath, sep="\t")
    
def wlen(essay):
    
    """Calculate average word length
    
    Input: essay (str)
    Output: average word length (float)
    """
    
    ws = str(essay).split()
    return sum(len(word) for word in ws) / len(ws)
    
    

def create_features(dataframe):
    """Main feature extraction
    
    Input: pandas dataframe
    Output: extracted features (pandas dataframe)"""
    
    # copy relevant columns    
    feas = dataframe[['essay_set','essay','domain1_score']].copy()
    
    # baseline
    feas['lemmacount'] = feas['essay'].apply(lc.calculate)
    feas['modifier2np'] = feas['essay'].apply(mc.calculate)
    feas['avglen'] = feas['essay'].apply(wlen)
    feas['mispell'] = feas['essay'].apply(sp.check_spell)
    
    # compute reading ease scores 
    feas['flesch'] = feas['essay'].apply(textstat.flesch_reading_ease)
    feas['smog'] = feas['essay'].apply(textstat.smog_index)
    feas['fleschkinc'] = feas['essay'].apply(textstat.flesch_kincaid_grade)
    feas['aread'] = feas['essay'].apply(textstat.automated_readability_index)
    feas['dalechall'] = \
				feas['essay'].apply(textstat.dale_chall_readability_score)
    feas['diffwords'] = feas['essay'].apply(textstat.difficult_words)
    feas['linwrite'] = feas['essay'].apply(textstat.linsear_write_formula)
    feas['gunning'] = feas['essay'].apply(textstat.gunning_fog)
    feas['textstand'] = feas['essay'].apply(textstat.text_standard)
				
    # getting binarized pos-ngrams    
    feas['pos'] = feas['essay'].apply(lc.poslist)
    cv = CountVectorizer(lowercase=False, ngram_range=(1,3), binary=True)
    ngs = cv.fit_transform(feas['pos'])
    posngrams = pd.DataFrame(ngs.toarray(), index=feas.index)
    
    # filter ngrams that occur less than 5 times
    ngcount = posngrams.sum(axis=0).to_frame()
    good_indices = ngcount[ngcount[0]>=5].index.values
    filtered_posngs = posngrams[good_indices]
    # convert to string
    filtered_posngs.rename(columns = lambda x: str(x), inplace=True)
    # join to features and drop POS feature
    feas = pd.concat([feas, filtered_posngs], axis=1)
    feas.drop('pos', axis=1, inplace=True)
    
    # compute LSA score    
    feas['lsascore'] = feas[['essay_set', 'essay']].apply(
    lsa.compute, axis=1)
    
    return feas
    
def save2file(dataframe):
    """Saves feature file
    
    Input: pandas dataframe"""
    pd.DataFrame.to_csv(dataframe, 'train_feas.csv')
    
if __name__ == '__main__':
    
    # compute and save features    
    df = read_data('data/training_set_rel3.tsv')
    feas = create_features(df)
    save2file(feas)
    
