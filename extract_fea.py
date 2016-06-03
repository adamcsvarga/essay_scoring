# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:59:25 2016

v0.1 -- added readability score features (June 2 2016)

@author: Adam Varga
"""

from textstat.textstat import textstat
import pandas as pd
#import lemma_count as lc, modifier_count as mc, spellcheck as sp

def read_data(filepath):
    return  pd.DataFrame.from_csv(filepath, sep="\t")
    
def wlen(essay):
    
    ws = str(essay).split()
    return sum(len(word) for word in ws) / len(ws)
    
    

def create_features(dataframe):
    feas = dataframe[['essay_set','essay','domain1_score']].copy()
    
    # already computed
    """feas['lemmacount'] = feas['essay'].apply(lc.calculate)
    feas['modifier2np'] = feas['essay'].apply(mc.calculate)
    feas['avglen'] = feas['essay'].apply(wlen)
    feas['mispell'] = feas['essay'].apply(sp.check_spell)"""
    
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

    
    return feas
    
def save2file(dataframe):
    pd.DataFrame.to_csv(dataframe, 'train_feas.csv')
    
if __name__ == '__main__':
    
    df = read_data('data/training_set_rel3.tsv')
    feas = create_features(df)
    save2file(feas)
    