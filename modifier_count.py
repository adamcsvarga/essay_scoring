# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:48:56 2016

Computation of average number of modifiers per NP

@author: Adam Varga
"""

import os, nltk
from nltk.parse import stanford

# set up Stanford parser path
nltk.internals.config_java(options='-xmx7G')
os.environ['JAVAHOME'] = 'C:\\Program Files\\Java\\jdk1.8.0_60\\bin'
os.environ['STANFORD_PARSER'] = \
'C:\\Users\\vurga\\AppData\\Roaming\\nltk_data\\stanford-parser-full-2015-04-20'
os.environ['STANFORD_MODELS'] = \
'C:\\Users\\vurga\\AppData\\Roaming\\nltk_data\\stanford-parser-full-2015-04-20'

def get_parse_trees(sentence_list):
    """Parse sentences
    
    Input: list of sentences (list of str)
    Output: list of parsed sentences"""
    p = stanford.StanfordParser()
    try:
        parsed_sentences = p.raw_parse_sents(sentence_list)
    
        return parsed_sentences
    except:
        return None
    
def traverse_and_count(tree):
    """Traverse through parse tree and count NPs and corresponding modifiers
    
    Input: parse tree
    Ouput: NP and modifier count (int, int)"""
    np_count = 0.0
    mod_count = 0.0
    strees = tree.subtrees()
    for stree in strees:
        if stree.label() == 'NP':
            np_count += 1
            np_subtrees = stree.subtrees()
            # JJ, PP or SBAR modifier in NP            
            for np_stree in np_subtrees:
                if np_stree.label() in ['JJ', 'PP', 'SBAR']:
                    mod_count += 1
            leaves_pos = stree.pos()
            ## NN NN type modifiers (immediate precedence)
            for i in range(0, len(leaves_pos)):
                if leaves_pos[i][1] == 'NN' and i < len(leaves_pos) - 1 \
                and leaves_pos[i+1][1] == 'NN':
                    mod_count += 1
                   
    
    return np_count, mod_count
            
    
def avg_mod_count(parsed_sentences):
    
    """Compute average number of modifiers per NP
    
    Input: parsed sentences
    Ouput: average modifier count per NP (float)"""
    
    np_count = 0.0
    mod_count = 0.0
    
    for sentence in parsed_sentences:
        for tree in sentence:
            np_plus, mod_plus = traverse_and_count(tree)
            np_count += np_plus
            mod_count += mod_plus
    if np_count == 0:
        return 0
    else:
        return mod_count / np_count
        
def calculate(essay):
    
    """Calculation for a given essay.
    
    Input: essay (str)
    Output: average modifier count per NP (float)"""
    
    sentences = nltk.sent_tokenize(str(essay))
    ps = get_parse_trees(sentences)
    if ps:    
        return avg_mod_count(ps)
    else:
        return 0
    
if __name__ == '__main__':
    
    # example test
    sentences = ["She's the woman with the hat.", "I saw the nice faculty woman."]
    ps = get_parse_trees(sentences)
    print(avg_mod_count(ps))