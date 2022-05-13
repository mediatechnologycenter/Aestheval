from sentiment_analysis.sent_clf import Predictor
from data.datasets import PCCD, Reddit
import json

SPLITS = ('train', 'val', 'test')
MIN_CHARS_THRESHOLD = 0

# Load datasets
pccd_dataset = {split: PCCD(split) for split in SPLITS}

# Load predictor
predictor = Predictor(model_path="cardiffnlp/twitter-roberta-base-sentiment-latest")

for name, dataset in pccd_dataset.items():
    for sample in dataset:
        
        #Initialize sentiment dictionary to None
        sentiments = {attribute: None for attribute in dataset.attributes}

        # Get all comments for the sample
        texts = [sample[attribute] for attribute in dataset.attributes]
        
        # We only one to predict the sentiment in those texts with a minimun length
        filtered_ids, filtered_texts = zip(*[(idx, text) for idx,  text in enumerate(texts) if len(texts) > MIN_CHARS_THRESHOLD])
        
        
        # Make predictions on filtered texts
        preds = predictor.predict(filtered_texts)

        # Assign
        for pred, id in zip(preds, filtered_ids):
            sentiments[dataset.attributes[id]] = pred

        sample['sentiments'] = sentiments

