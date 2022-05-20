# Image Aesthetic Assessment Baselines

* [NIMA: Neural IMage Assessment](https://ieeexplore.ieee.org/document/8352823)
* [Effective Aesthetics Prediction with Multi-level Spatially Pooled Features (MLSP)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hosu_Effective_Aesthetics_Prediction_With_Multi-Level_Spatially_Pooled_Features_CVPR_2019_paper.pdf)


## ViT

Then, install the timm library (the following version should be used to use our pre-trained weights).
```
pip install git+https://github.com/rwightman/pytorch-image-models.git@95feb1da41c1fe95ce9634b83db343e08224a8c5
```

In timm, variable length input images are not supported. Replace the file *vision_transformer.py* of this repo to where you installed timm by:

```
cp aestheval/baselines/model/ViT.py ~/anaconda3/envs/aestheval/lib/python3.10/site-packages/timm/models/vision_transformer.py
```


## Acknoledments
We would like to thank:
* [yunxiaoshi](https://github.com/yunxiaoshi) for providing the PyTorch implementation of NIMA method: https://github.com/yunxiaoshi/Neural-IMage-Assessment
* [Vlad Hosu](https://github.com/subpic) for the implementation of the MLSP method: https://github.com/subpic/ava-mlsp
* TODO: Add timm ack