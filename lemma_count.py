# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 09:52:46 2016

Lemma counter for essay scoring

Usage: python lemma_count.py <input_file>
Input: text file containing English text
Output: number of unique lemmas

Requires NLTK

@author: Adam Varga
"""
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
import sys, nltk, re

def read_file(infile):
    """ Reads input file line by line
    Input: filename
    Output: list of lines"""
    
    try:
        with open(infile, 'r') as source: lines = source.readlines()
        return lines
    except IOError:
        print("Input file not found: " + infile)
        exit(1)

def tokenize(line):
    """ Tokenizes  a string
    Input: string to be tokenized
    Output: list of tokens """
    
    orig = line.strip()
    return nltk.word_tokenize(orig)

def pos(tokens):
    """ PoS tags a list of tokens
    Input: list of tokens to be PoS-tagged
    Output: list of tuples with (token, PoS-tag)"""
    
    return nltk.pos_tag(tokens)
    
def penn_2_wordnet(pos_tag):
    """ Converts Penn PoS-tags to Wordnet-style pos tags
    Input: Penn pos tag
    Output: wordnet Pos tag"""
    
    if pos_tag.startswith('N'): 
        return wordnet.NOUN
    elif pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    # eliminate punctuation
    elif re.match('[\.:,\(\)]', pos_tag):
        return None
    # default to noun
    else:
        return wordnet.NOUN
    
def get_lemmas(token_tuples):
    """ Lemmatizes a list of token-pos tuples
    Input: list of token-pos tuples using the Penn tagset
    Output: list of tokens"""
    
    lemmas = []
    wordnet_lemmatizer = WordNetLemmatizer()
    
    for token_tuple in token_tuples:
        # convert penn pos-tags to wordnet pos-tags
        pos = penn_2_wordnet(token_tuple[1])
        if pos: 
            lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
        
    return lemmas
    
def count_lemmas(lemma_list):
    """ Count number of unique lemmas
    Input: list of lemmas
    Output: number of unique lemmas"""
    
    return len(set(lemma_list))
    
    
def calculate(essay):
    
    lemma_list = []
    #token_count = 0.0
    
    tokens = tokenize(str(essay).lower())
    #token_count = len(tokens)
    posd = pos(tokens)       
    lemmas = get_lemmas(posd)
    lemma_list += lemmas
    
    lcount = count_lemmas(lemma_list)
    #return float(lcount) / token_count
    return lcount
    
def poslist(essay):
    tokens = tokenize(str(essay).lower())
    poslist = []
    p = pos(tokens)
    for pp in p:
        poslist.append(pp[1])
        
    return ' '.join(poslist)
    
def comma_count(essay):
    
    return poslist(essay).count(',')
				
def posngram(essay):
	
	#POS-tag essay
	tokens = tokenize(str(essay).lower())
	posd = pos(tokens)
	
	# unigrams
	ugs = set()
	for p in posd:
		ugs.add(p[1])
		
	# bigrams
	bgs = set()
	for i in range(0, len(posd)-1):
		bi = (posd[i][1], posd[i+1][1])
		bgs.add(bi)
		
	# trigrams
	tgs = set()
	for i in range(0, len(posd)-2):
		tgs.add((posd[i][1], posd[i+1][1], posd[i+2][1]))
		
	return ugs, bgs, tgs
    
    
if __name__ == '__main__':
    try:
        lines = read_file(sys.argv[1])
    except IndexError:
        print("Usage: python lemma_count.py <input_file>")
        exit(1)    
    
    lemma_list = []
    token_count = 0.0
    for line in lines:
        tokens = tokenize(line.lower())
        token_count += len(tokens)
        posd = pos(tokens)       
        lemmas = get_lemmas(posd)
        lemma_list += lemmas
   
    lcount = count_lemmas(lemma_list)
    nlcount = float(lcount) / token_count
    print("Unique lemma count: ", count_lemmas(lemma_list))
    print("Normalized lemma count: ", str(nlcount))
    

    
    


