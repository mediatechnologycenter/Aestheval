from aestheval.sentiment_analysis.sent_clf import Predictor
from aestheval.data.datasets import PCCD, Reddit, AVADataset
import json
from tqdm import tqdm
import statistics as st

SPLITS = ('train', 'validation', 'test')
MIN_CHARS_THRESHOLD = 0

def compute_score(sentiments: "list[dict]"):
    #TODO: This is a base approach, we have to take into account images with different number of comments in an smarter way
    
    # Compute score for the 7 comments each image might have. Compute only if prediction is not None.
    scores = [attribute['Positive'] + 0.5*attribute['Neutral'] for attribute in sentiments.values() if attribute]
    
    stdev = 0
    # variance requires at least two data points 
    if len(scores) > 1:
        stdev = st.stdev(scores)
    
    return st.mean(scores), stdev, len(scores)


def sentiment_pccd():
    # Load datasets
    pccd_dataset = {split: PCCD(split) for split in SPLITS}

    # Load predictor
    predictor = Predictor(model_path="cardiffnlp/twitter-roberta-base-sentiment-latest")

    for name, dataset in pccd_dataset.items():
        for img, data in tqdm(dataset):
            
            #Initialize sentiment dictionary to None
            sentiments = {attribute: None for attribute in dataset.attributes}

            # Get all comments for the sample
            texts = [data[attribute] for attribute in dataset.attributes]
            
            # We only one to predict the sentiment in those texts with a minimun length
            filtered_ids, filtered_texts = zip(*[(idx, text) for idx,  text in enumerate(texts) if len(text) > MIN_CHARS_THRESHOLD])
            
            
            # Make predictions on filtered texts
            preds = predictor.predict(filtered_texts)

            # Assign predictions to their corresponding attribute
            for pred, id in zip(preds, filtered_ids):
                sentiments[dataset.attributes[id]] = pred
            data['sentiment'] = sentiments
            data['mean_score'], data['stdev_score'], data['number_of_scores'] = compute_score(sentiments)

    for name, dataset in pccd_dataset.items():
        processed = [data for img, data in dataset]
        with open(f'data/PCCD/processed_{name}.json', 'w') as f:
            json.dump(processed, f, indent=1)

def sentiment_reddit():
    # Load datasets
    reddit_dataset = {split: Reddit(split) for split in SPLITS}

    # Load predictor
    predictor = Predictor(model_path="cardiffnlp/twitter-roberta-base-sentiment-latest")

    for name, dataset in reddit_dataset.items():
        for img, data in tqdm(dataset):
            

            comments = data['first_level_comments_values']
            
            #Initialize sentiment dictionary to None
            sentiments = {idx: None for idx in range(len(comments))}
            
            # We only one to predict the sentiment in those texts with a minimun length
            filtered_ids, filtered_texts = zip(*[(idx, text) for idx,  text in enumerate(comments) if len(text) > MIN_CHARS_THRESHOLD])
            
            
            # Make predictions on filtered texts
            preds = predictor.predict(filtered_texts)

            # Assign predictions to their corresponding attribute
            for pred, id in zip(preds, filtered_ids):
                sentiments[id] = pred
            data['sentiment'] = sentiments
            data['mean_score'], data['stdev_score'], data['number_of_scores'] = compute_score(sentiments)

    for name, dataset in reddit_dataset.items():
        processed = [data for img, data in dataset]
        with open(f'data/reddit/processed_{name}.json', 'w') as f:
            json.dump(processed, f, indent=1)

def sentiment_ava():
    # Load datasets
    ava_dataset = {split: AVADataset(split) for split in SPLITS}

    # Load predictor
    predictor = Predictor(model_path="cardiffnlp/twitter-roberta-base-sentiment-latest")

    for name, dataset in ava_dataset.items():
        for img, data in tqdm(dataset):
            

            comments = data['comments']
           
            #Initialize sentiment dictionary to None
            sentiments = {idx: None for idx in range(len(comments))}
            
            # We only one to predict the sentiment in those texts with a minimun length
            filtered_ids, filtered_texts = zip(*[(idx, text) for idx,  text in enumerate(comments) if len(text) > MIN_CHARS_THRESHOLD])
            
            
            # Make predictions on filtered texts
            preds = predictor.predict(filtered_texts)

            # Assign predictions to their corresponding attribute
            for pred, id in zip(preds, filtered_ids):
                sentiments[id] = pred
            data['sentiment'] = sentiments
            data['mean_score'], data['stdev_score'], data['number_of_scores'] = compute_score(sentiments)

    for name, dataset in ava_dataset.items():
        processed = [data for img, data in dataset]
        with open(f'data/ava/processed_{name}.json', 'w') as f:
            json.dump(processed, f, indent=1)

if __name__ == "__main__":
    # sentiment_pccd()
    # sentiment_reddit()
    sentiment_ava()

