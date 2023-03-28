from aestheval.text_prediction.predictor import Predictor
from aestheval.data.datasets import PCCD, Reddit, AVA
from aestheval.text_prediction.informativeness_score import all_the_steps, compute_informativeness_score
import json
from tqdm import tqdm
import statistics as st
import os
import numpy as np
from collections import Counter

SPLITS = ('train', 'validation', 'test')
MIN_CHARS_THRESHOLD = 0


def info_pccd(root_dir):
    # Load datasets
    pccd_dataset = {split: PCCD(split, dataset_path=os.path.join(root_dir,'PCCD'), load_images=False) for split in SPLITS}

    unigram_dictionary = {}
    bigram_dictionary = {}

    # Load predictor

    for name, dataset in pccd_dataset.items():
        for img, data in tqdm(dataset):
            

            # Get all comments for the sample
            texts = [data[attribute] for attribute in dataset.attributes]
                       
            
            # Make predictions on filtered texts
            comments = [{'raw': text} for text in texts]
            reduced_tokenized_comments = list(filter(lambda x: all_the_steps(x, unigram_dictionary, bigram_dictionary), comments))
            data['reduced_tokenized_comments'] = reduced_tokenized_comments

    unigram_scores = dict(zip(list(unigram_dictionary.keys()), np.array(list(unigram_dictionary.values()))/float(np.sum(list(unigram_dictionary.values())))))
    bigram_scores = dict(zip(list(bigram_dictionary.keys()), np.array(list(bigram_dictionary.values()))/float(np.sum(list(bigram_dictionary.values())))))


    for name, dataset in pccd_dataset.items():
        for img, data in tqdm(dataset):

            # Make predictions on filtered texts
            info_scores = [compute_informativeness_score(comment, unigram_scores, bigram_scores) for comment in data['reduced_tokenized_comments']]        

            data['info_scores'] = info_scores

    for name, dataset in pccd_dataset.items():
        processed = [data for img, data in dataset]
        with open(os.path.join(root_dir, f'PCCD/processed_info_{name}.json'), 'w') as f:
            json.dump(processed, f, indent=1)

def info_reddit(root_dir):
    # Load datasets
    reddit_dataset = {split: Reddit(split, dataset_path=os.path.join(root_dir,'RPCD'), load_images=False) for split in SPLITS}


    unigram_dictionary = {}
    bigram_dictionary = {}


    for name, dataset in reddit_dataset.items():
        for img, data in tqdm(dataset):
            

            comments = data['first_level_comments_values']
            
            # Make predictions on filtered texts
            comments = [{'raw': text} for text in comments]
            reduced_tokenized_comments = list(filter(lambda x: all_the_steps(x, unigram_dictionary, bigram_dictionary), comments))
            data['reduced_tokenized_comments'] = reduced_tokenized_comments

    unigram_scores = dict(zip(list(unigram_dictionary.keys()), np.array(list(unigram_dictionary.values()))/float(np.sum(list(unigram_dictionary.values())))))
    bigram_scores = dict(zip(list(bigram_dictionary.keys()), np.array(list(bigram_dictionary.values()))/float(np.sum(list(bigram_dictionary.values())))))

    for name, dataset in reddit_dataset.items():
        for img, data in tqdm(dataset):

            # Make predictions on filtered texts
            info_scores = [compute_informativeness_score(comment, unigram_scores, bigram_scores) for comment in data['reduced_tokenized_comments']]        

            data['info_scores'] = info_scores
            

    for name, dataset in reddit_dataset.items():
        processed = [data for img, data in dataset]
        with open(os.path.join(root_dir,f'RPCD/processed_info_{name}.json'), 'w') as f:
            json.dump(processed, f, indent=1)

def info_ava(root_dir):
    # Load datasets
    ava_dataset = {split: AVA(split,  dataset_path=os.path.join(root_dir,'ava'),  load_images=False) for split in SPLITS}

    unigram_dictionary = {}
    bigram_dictionary = {}

    for name, dataset in ava_dataset.items():
        for img, data in tqdm(dataset):
            

            comments = data['comments']
           
            comments = [{'raw': text} for text in comments]
            reduced_tokenized_comments = list(filter(lambda x: all_the_steps(x, unigram_dictionary, bigram_dictionary), comments))
            data['reduced_tokenized_comments'] = reduced_tokenized_comments
    
    unigram_scores = dict(zip(list(unigram_dictionary.keys()), np.array(list(unigram_dictionary.values()))/float(np.sum(list(unigram_dictionary.values())))))
    bigram_scores = dict(zip(list(bigram_dictionary.keys()), np.array(list(bigram_dictionary.values()))/float(np.sum(list(bigram_dictionary.values())))))

    for name, dataset in ava_dataset.items():
        for img, data in tqdm(dataset):

            # Make predictions on filtered texts
            info_scores = [compute_informativeness_score(comment, unigram_scores, bigram_scores) for comment in data['reduced_tokenized_comments']]        

            data['info_scores'] = info_scores

    for name, dataset in ava_dataset.items():
        processed = [data for img, data in dataset]
        with open(os.path.join(root_dir,f'ava/processed_info_{name}.json'), 'w') as f:
            json.dump(processed, f, indent=1)

