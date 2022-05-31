# Prepare data

Note: every preprocessed file or preextracted features can be found in [link](https://drive.google.com/open?id=1eCdz62FAVCGogOuNhy87Nmlo5_I0sH2J).

## Reddit

Run:

```bash
$ python scripts/prepro_labels_reddit.py --input_json ../../data/reddit/ --output_json data/reddit.json --output_h5 data/reddit
```

`prepro_labels_reddit.py` will map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data/reddit.json` and discretized caption data are dumped into `data/reddit_label.h5`.

### Image features: Resnet features

Download pretrained resnet models. The models can be downloaded from [here](https://drive.google.com/open?id=0B7fNdx_jAqhtbVYzOURMdDNHSGM), and should be placed in `data/imagenet_weights`.

Then:

```bash
python scripts/prepro_feats_reddit.py --input_json ../../data/reddit --output_dir data/reddit --images_root ../../data/reddit --model_root imagenet_weights
```

`prepro_feats_reddit.py` extract the resnet101 features (both fc feature and last conv feature) of each image. The features are saved in `data/reddit_fc` and `data/reddit_att`.

(Check the prepro scripts for more options, like other resnet models or other attention sizes.)