#!/bin/bash

python -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --train-data="csvs/clip_train.csv" \
    --val-data="csvs/clip_val.csv" \
    --csv-img-key "latent_image_path" \
    --csv-caption-key "prompt" \
    --warmup 10000 \
    --batch-size=32 \
    --lr=1e-6 \
    --wd=0.1 \
    --epochs=30 \
    --workers=8 \
    --model ViT-B-16 \
    --pretrained datacomp_xl_s13b_b90k