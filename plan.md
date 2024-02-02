On fine-tuned CLIP, either:
1. Finetune for downstream classification task using wise-ft
or
2. Train a simple regression feedforward network with a bunch of linear layers on CLIP extracted embeddings the same way it's done in aesthetic-scorer

debug train in wandb btw

-----
possible things to improve:
- redo the whole thing with tiny autoencoder decoding and classifiction on image space
- Finetuning the finetuned CLIP for downstream classification task using wise-ft
- Using different scheduler in forward process to construct dataset, since the training data was generated using different schedulers.
- Construct dataset with noise features on different timesteps, not only on fixed timestep like 10
- Add random noise on each step of forward diffusion process, unlike the stadard image generation process
- Using greater batch size for CLIP training, would probably improve a lot since loss depends on batch size
- (perhaps, somehow fine-tuned CLIP somehow learned to ignore the artifacts on images, therefore finetuning it won't do much, but ill rethink this statement when 
results of either of two methods listed on top won't work)
- ALSO USE BICUBIC APPROXIMATION FOR CLIP TRAINING RESAMPLING (I DONT KNOW WHAT IS THE STOCK RESAMPLER)
- Use idea of calculation of CDF on each diffusion step and summing the probabilities until threshold reached, but would require insane amount of compute probably,
derived from PALBERT