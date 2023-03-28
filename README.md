# Image aesthetic assessment framework

This repository makes easy to access and process different datasets used for image aesthetic assessment methods, as well as the newly introduced Reddit Photo Critique Dataset (RPCD). It contains the framework for running the experiments described in the paper '*Understanding Aesthetics with Language: A Photo Critique Dataset for Aesthetic Assessment*'.

## RPCD: Reddit Photo Critique Dataset
The Reddit Photo Critique Dataset (RPCD) contains tuples of image and photo critiques. It consists of 74K images and 220K comments and is collected from a Reddit community used by hobbyists and professional photographers to improve their photography skills by leveraging constructive community feedback.

The proposed dataset differs from previous aesthetics datasets mainly in three aspects, namely:

* the large scale of the dataset and the extension of the comments criticizing different aspects of the image;
* it contains mostly UltraHD images;
* it can easily be extended to new data as it is collected through an automatic pipeline.

A detailed description of the dataset can be found on the [datasheet](data/datasheet.md).

## Models

ViT L/16 models trained on AVA, PCCD and RPCD datasets to predict the proposed sentiment score can be found [here](https://drive.google.com/drive/folders/1KuuiyNJUa92rCUtv9JO6gyAJ0plpqxcS?usp=sharing).

## Steps
The files available in [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6656802.svg)](https://zenodo.org/record/6985507) correspond to the ids of the posts, divided in the splits used in the experiments of our work.

## Steps for running the experiments

### 1. Create environment and install requirements

```
conda env create -n aestheval
pip install -e .
```

### 2. Download datasets

Follow the [data README](https://github.com/mediatechnologycenter/aestheval/tree/main/data) to download the RPCD dataset.

### 3. Process RPCD

```
python aestheval/data/reddit/prepare_dataset.py
```

Default `--reddit_dir` is `data/`.

### 4. Predict sentiment of comments and compute informativeness score

On RPCD dataset (takes a while)

```
python main.py --compute_sentiment_score --compute_informativeness_score --dataset_name Reddit
```

Already processed files can be found under `data/`. This directory can be changed using the `--data_path` argument:

- ``processed_info_test.json`` 
- ``processed_info_train.json`` 
- ``processed_info_validation.json``

## Repository structure

The repository is structured as follows:
- `EDAs`: Exploratory Data Analysis of AVA, DPC, PCCD and RPCD, as well as the concatenation of them all.
- `aestheval`: Library with the data utils to download, load and process data; as well with the baselines used in this project.
- `results`: Results of the expeeriments with Aesthetic ViT, ViT + Linear probe and NIMA.
- `data`: Default directory for the data.


## Citation
If you use this repository, please cite the following paper:
* Daniel Vera Nieto, Luigi Celona, and Clara Fernandez-Labrador. "Understanding Aesthetics with Language: A Photo Critique Dataset for Aesthetic Assessment." In *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks* (2022) [[PDF]](https://openreview.net/forum?id=-VyJim9UBxQ).

```bibtex
@inproceedings{nieto2022understanding,
    title={Understanding Aesthetics with Language: A Photo Critique Dataset for Aesthetic Assessment},
    author={Daniel Vera Nieto and Luigi Celona and Clara Fernandez Labrador},
    booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2022},
    url={https://openreview.net/forum?id=-VyJim9UBxQ}
}
```
