# aestheval

## 1. Create environment

TODO: Improve env setup

```
conda env create -f environment.yml
pip install -e .
```

## 2. Download datasets

```
python main.py --download_data
```

## 3. Predict sentiment of comments 

On PCCD, AVA and Reddit (takes a while)

```
python main.py --compute_sentiment_score
```

Already processed files can be found under `data/`. This directory can be changed using the `--data_path` argument


## Future work
 - [ ] Optimize sequential sentiment score computation
 - [ ] Add ViT training code
 - [ ] Standarize datasets: I don't like how the load of the processed dataset (with sentiment scores) is managed.
 - [ ] Accuracy metrics is not really useful in the whole library, since it's arbitrarily defined and it should be defined differently for each dataset
 - [ ] Upload aesthetic aspects classifier
 - [ ] Filtering by num words should be improved
 - [ ] Integrate informativeness score properly