# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:09:59 2016

Counts mispelled words

@author: Adam Varga
"""

import re, collections

# return all words
def words(text): return re.findall('[a-z]+', text.lower()) 

def train(features):
    """Trains spelling checker on word list
    
    Input: list of words (list of str)
    Output: dictionary with word counts (dict {str:int})"""
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

# known words
NWORDS = train(words(open('data/big.txt').read()))

def known(word):
    """Get known words"""
    return word in NWORDS
    
def check_spell(essay):
    """Performs spell checking on essay against known word list
    
    Input: essay (string)
    Ouput: number of misspelled words (int)"""
    mispell_count = 0
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
    clean_essay = re.sub(r'num ', 'number', clean_essay)
    
    ws = clean_essay.split()
    for word in ws:
        if not known(word):
            mispell_count += 1
            
    return mispell_count
    
if __name__ == '__main__':
    print(known("num"))