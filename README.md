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
 [] Optimize sequential sentiment score computation
 [] Add ViT training code