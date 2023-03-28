"""
code adapted from https://github.com/V-Sense/Aesthetic-Image-Captioning-ICCVW-2019/blob/master/clean_using_subjectivity.py

@article{ghosal2019aesthetic,
  title={Aesthetic Image Captioning From Weakly-Labelled Photographs},
  author={Ghosal, Koustav and Rana, Aakanksha and Smolic, Aljosa},
  journal={arXiv preprint arXiv:1908.11310},
  year={2019}
}
"""

'''
1. remove non-english captions
2. Truncates woooooooow to woow, and what!!!!!!!!!!!!!! to what!
3. Puncuations except ?, !, ,, . are removed
4. Tokenizes into unigrams and bigrams
5. Computes probabilities based on corpus freq and filters captions
6. Also coments with manually selected bad words are removed by hard coding zero probabilities
'''
# from __future__ import print_function
import json
import re
import io
from random import shuffle
import pdb
from nltk.tokenize import RegexpTokenizer
from string import digits, punctuation
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.util import ngrams
from collections import Counter
import numpy as np
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
import itertools

#exclude = set(punctuation + digits) - set(['!','?', '.', ',','\''])
#comma removed for comparing with PCCD
exclude = set(punctuation + digits) - set(['!','?', '.', '\'',','])
tokenizer = RegexpTokenizer(r'\w+\S*\w*')
stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

subjectivity_threshold = 120
objectivity_threshold = 20

remove_char_map = dict((ord(char), None) for char in exclude)
lame_word_list = ['challenge', 'challenges', 'congrats', 'congratulations',\
'congratulation', 'title','titles',  'ribbon', 'ribbons','score', 'scores','scored', \
 'comment', 'comments', 'commented','favorites', 'favorite', 'fav','thanks', 'thank', 'vote', 'voting',\
 'votes', 'voters', 'voter','voted', 'entry', 'entries', 'dpc', 'dpchallenge', 'award', 'awards', 'critique', 'rating', 'luck', 'theme']

replace_char_map = {
    '\xc2\x82' : ',',        # High code comma
    '\xc2\x84' : ',,',       # High code double comma
    '\xc2\x85' : '...',      # Tripple dot
    '\xc2\x88' : '^',        # High carat
    '\xc2\x91' : '\x27',     # Forward single quote
    '\xc2\x92' : '\x27',     # Reverse single quote
    '\xc2\x93' : '\x22',     # Forward double quote
    '\xc2\x94' : '\x22',     # Reverse double quote
    '\xc2\x95' : ' ',
    '\xc2\x96' : '-',        # High hyphen
    '\xc2\x97' : '--',       # Double hyphen
    '\xc2\x99' : ' ',
    '\xc2\xa0' : ' ',
    '\xc2\xa6' : '|',        # Split vertical bar
    '\xc2\xab' : '<<',       # Double less than
    '\xc2\xbb' : '>>',       # Double greater than
    '\xc2\xbc' : '1/4',      # one quarter
    '\xc2\xbd' : '1/2',      # one half
    '\xc2\xbe' : '3/4',      # three quarters
    '\xca\xbf' : '\x27',     # c-single quote
    '\xcc\xa8' : '',         # modifier - under curve
    '\xcc\xb1' : '',          # modifier - under line
    '\xc2\xb4' : '\''
    
}

print_flag_array = [True] * 2 + [False] * 8
shuffle(print_flag_array)

unigram_dictionary = {}
bigram_dictionary = {}

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


def reduce_lengthening_word(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

def reduce_lengthening_comment(comment: "dict[str, str]"):
    comment['clean'] = ' '.join(list(map(reduce_lengthening_word, comment['clean'].split())))
    return comment
    
def strip_consecutive_punctutaion(comment: "dict[str, str]"):
    mul_punc = re.compile(r'([.,/#!$%^&*;:{}=_`~()-?])[.,/#!$%^&*;:{}=_`~()-?]+')    
    comment['clean'] = mul_punc.sub(r'\1', comment['clean'])
    return comment
    
def clean_string(comment: "dict[str, str]"):
    # print(comment)
    low_text = comment['raw'].lower()
    replaced_text = low_text.translate(replace_char_map)
    comment['clean'] = replaced_text.translate(remove_char_map)    
    return comment

def update_unigram_dictionary(unigram, unigram_dictionary):
    pos = unigram[1]
    if pos in ['NN', 'NNS']:
        try:
           unigram_dictionary[unigram[0]] += 1
        except KeyError as e:
            unigram_dictionary[unigram[0]] = 1
        return True
    else:
        return False
    
def update_bi_gram_dictionary(bigram, bigram_dictionary):
    if bigram[0][1] in ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'] \
    and bigram[1][1] in ['NN', 'NNS', 'JJ', 'JJR', 'JJS']:
        bi_gram_word = bigram[0][0] + '_' +bigram[1][0]
        try:
           bigram_dictionary[bi_gram_word] += 1
        except KeyError as e:
            bigram_dictionary[bi_gram_word] = 1
        return True        
    else:
        return False
    
def tokenize(comment: "dict[str, str]"):
    tokens = re.findall(r"[\w']+|[.,!?;]", comment['clean'], re.UNICODE)
    token_pos = list(pos_tag(tokens))
    comment['tokens'] = token_pos
    return comment

def update_dicts(comment, unigram_dictionary, bigram_dictionary):
    token_pos = comment['tokens']
    unigrams =  [(i,j) for i,j in token_pos if i not in (stop | set(punctuation))]
    bigrams = ngrams(unigrams, 2)
    filtered_unigrams = list(filter(lambda x: update_unigram_dictionary(x, unigram_dictionary), unigrams))
    filtered_bigrams = list(filter(lambda x: update_bi_gram_dictionary(x, bigram_dictionary), bigrams))
    comment['unigrams'] = filtered_unigrams
    comment['bigrams'] = filtered_bigrams
    
       
def all_the_steps(comment: "dict[str, str]", unigram_dictionary, bigram_dictionary) -> bool:
      punc_dig_free_comment = clean_string(comment)
      multiple_punc_removed = strip_consecutive_punctutaion(punc_dig_free_comment)
      reduced_comment = reduce_lengthening_comment(multiple_punc_removed)
      tokenized_comment = tokenize(reduced_comment)
      if remove_dpc(tokenized_comment):      
          update_dicts(tokenized_comment, unigram_dictionary, bigram_dictionary)
          if len(tokenized_comment['unigrams']) != 0 or len(tokenized_comment['bigrams']) != 0:
              return True
          else:
              return False
      else:
          return False
      
def remove_dpc(comment: "dict[str, str]"):
    def lemmatize(pos):
        global lemmatizer
        if pos[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            return (lemmatizer.lemmatize(pos[0], wordnet.NOUN), pos[1])
        elif pos[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            return (lemmatizer.lemmatize(pos[0], wordnet.VERB), pos[1])
        elif pos[1] in ['JJ', 'JJR', 'JJS']:
            return (lemmatizer.lemmatize(pos[0], wordnet.ADJ), pos[1])        
        elif pos[1] in [ 'RB', 'RBR', 'RBS']:
            return (lemmatizer.lemmatize(pos[0], wordnet.ADV), pos[1])
        else:
            return pos    
    token_pos = comment['tokens']
    lemmatized_tokens = list(map(lemmatize, token_pos))
    for token in lemmatized_tokens:
        if token[0] in lame_word_list:
            return False
    comment['tokens'] = lemmatized_tokens
    return True


def compute_informativeness_score(comment: "dict[str, str]", unigram_scores, bigram_scores) -> float:
    global subjectivity_threshold, objectivity_threshold, print_flag_array
    unigram_score = 1.0
    bigram_score = 1.0
    too_objective = True
    too_subjective = True
    print_flag = print_flag_array[np.random.randint(10)]

    for unigram in comment['unigrams']:
        unigram_score *= unigram_scores[unigram[0]]
    for bigram in comment['bigrams']:
        bigram_score *= bigram_scores[bigram[0][0] +'_' +bigram[1][0]]
    
    informativeness_score = -np.log(unigram_score * bigram_score)/2

    if informativeness_score <= subjectivity_threshold:
        too_subjective = False
    if informativeness_score >= objectivity_threshold and len(comment['tokens']) >= 5:
        too_objective = False

    final_flag = not (too_subjective or too_objective )

    if not final_flag:
        if too_objective and print_flag:
            print(("{:0.1e}".format(bigram_score)), \
            ("{:0.1e}".format(unigram_score)), \
            ("{:0.1f}".format(-np.log(unigram_score * \
            bigram_score)/2)) , comment['clean'])
        else :
            if print_flag:
                print(("{:0.1e}".format(bigram_score)),\
                ("{:0.1e}".format(unigram_score)), \
                ("{:0.1f}".format(-np.log(unigram_score * \
                bigram_score)/2)), comment['clean'])
    else:
        if print_flag:
            print(("{:0.1e}".format(bigram_score)),\
            ("{:0.1e}".format(unigram_score)),\
            ("{:0.1f}".format(-np.log(unigram_score * \
                bigram_score)/2)), comment['clean'])

    return informativeness_score