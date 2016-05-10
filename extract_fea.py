# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:59:25 2016

@author: vurga
"""

from nltk.corpus import words
import pandas as pd, lemma_count as lc, modifier_count as mc, spellcheck as sp

def read_data(filepath):
    return  pd.DataFrame.from_csv(filepath, sep="\t")
    
def wlen(essay):
    
    ws = str(essay).split()
    return sum(len(word) for word in ws) / len(ws)
    
    

def create_features(dataframe):
    feas = dataframe[['essay_set','essay','domain1_score']].copy()
    
    feas['lemmacount'] = feas['essay'].apply(lc.calculate)
    feas['modifier2np'] = feas['essay'].apply(mc.calculate)
    feas['avglen'] = feas['essay'].apply(wlen)
    feas['mispell'] = feas['essay'].apply(sp.check_spell)
    
    return feas
    
def save2file(dataframe):
    pd.DataFrame.to_csv(dataframe, 'train_feas.csv')
    
if __name__ == '__main__':
    
    df = read_data('data/training_set_rel3.tsv')
    feas = create_features(df)
    save2file(feas)
    