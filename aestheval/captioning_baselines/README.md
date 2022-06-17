
## BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

This is our baseline for Aesthetic Image Captioning (AIC). To finetune the COCO pre-trained model for AIC, from the `BLIP` directory run:
``python train_caption.py --config configs/caption_<dataset_name>.yaml --output_dir output/Caption_<dataset_name>``,
where `dataset name` can be ava, pccd, or reddit.

See `BLIP/README.md` for further details.