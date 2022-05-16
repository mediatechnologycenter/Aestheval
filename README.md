# aestheval

## Create environment

TODO: Improve env setup

1.
```
conda env create -f environment.yml
```

```
pip install -e .
```

## Download datasets

```
python aestheval/data/data-tools/data_downloader.py
```


## Predict sentiment of comments 

On PCCD, AVA and Reddit (takes a while)

```
python main.py
```

Already processed files can be found under `data/`