# aestheval

This repo allows make easy to access and process different datasets used usually for aesthetic assessment methods, as well as the newly introduced Reddit Photo Critique Dataset.

## Get Reddit dataset
Zenodo: https://zenodo.org/record/6656802#.YqyS6xuxWhA
HF datasets: Coming soon...

Files in the dataset:

- ``reddit_photocritique_posts.pkl`` - 260.7 MB - Processed posts of Reddit
- ``raw_reddit_photocritique_comments.csv`` - 279.0 MB - File with allt he comments belonging to every post. Not processed.
- ``processed_info_test.json`` - 208.6 MB - Test set as used for trainings and analysis	
- ``processed_info_train.json`` - 725.7 MB 	- Train set as used for trainings and analysis
- ``processed_info_validation.json`` - 104.7 MB - Validation set as used for trainings and analysis



## Steps

### 1. Create environment

TODO: Improve env setup

```
conda env create -f environment.yml
pip install -e .
```

### 2. Download datasets

```
python main.py --download_data
```

### 3. Process Reddit dataset

python aestheval/data/reddit/prepare_dataset.py --only_predictions  # Reads the provided dataframe, the original downloaders will be provided soon

### 4. Predict sentiment of comments and compute informativeness score

On PCCD, AVA and Reddit (takes a while)

```
python main.py --compute_sentiment_score --compute_informativeness_score
```

Already processed files can be found under `data/`. This directory can be changed using the `--data_path` argument



## Cite

## Future work
 - [ ] Optimize sequential sentiment score computation
 - [ ] Add ViT training code
 - [ ] Standarize datasets: I don't like how the load of the processed dataset (with sentiment scores) is managed.
 - [ ] Accuracy metrics is not really useful in the whole library, since it's arbitrarily defined and it should be defined differently for each dataset
 - [ ] Upload aesthetic aspects classifier
 - [ ] Filtering by num words should be improved
 - [ ] Integrate informativeness score properly
 - [ ] Add datasets downloaders
