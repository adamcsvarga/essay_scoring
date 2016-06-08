# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:09:59 2016

@author: vurga
"""

import re, collections

def words(text): return re.findall('[a-z]+', text.lower()) 

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

NWORDS = train(words(open('data/big.txt').read()))

def known(word):
    return word in NWORDS
    
def check_spell(essay):
    
    mispell_count = 0
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
    clean_essay = re.sub(r'num ', 'number', clean_essay)
    
    ws = clean_essay.split()
    for word in ws:
        if not known(word):
            mispell_count += 1
           # print(word)
            
    return mispell_count
    
if __name__ == '__main__':
    print(known("num"))