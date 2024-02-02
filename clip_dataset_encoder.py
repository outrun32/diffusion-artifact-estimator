import torch
import open_clip
from PIL import Image
import pandas as pd 
from tqdm import tqdm
import os




#The image embeddings (projected embeddings) obtained by applying the projection layer to the pooler_output. From HF docs
def encode_dataframe(clip_hf_path, df, filename="ViT-B-16-finetuned-latent-embeds", clip_batch_size=128):
    df = pd.read_parquet('parquets/latents-approximated.parquet')
    embedding_file_path = f"parquets/{filename}.parquet"
    model, _, preprocess = open_clip.create_model_and_transforms(clip_hf_path)
    model.to('cuda')
    if os.path.exists(embedding_file_path):
        result_df = pd.read_parquet(embedding_file_path)
    else:
        result_df = pd.DataFrame(columns=["latent_image_path", "embedding"])
    missing_image_names = df[~df["latent_image_path"].isin(result_df["latent_image_path"])]["latent_image_path"].unique()
    print(f"Missing {len(missing_image_names)} embeddings...")
    for pos in tqdm(range(0, len(missing_image_names), clip_batch_size)):
        image_paths = missing_image_names[pos:pos+clip_batch_size]
        pil_images = [Image.open(image_path) for image_path in image_paths]
        inputs = torch.stack([preprocess(pil_image) for pil_image in pil_images], dim=0).to('cuda')
        with torch.no_grad(), torch.cuda.amp.autocast():
            embeddings = model.encode_image(inputs)
            result_df = pd.concat([result_df, pd.DataFrame({"latent_image_path": image_paths, "embedding": list(embeddings.cpu().numpy())})], ignore_index=True) #might detach too
    result_df.to_parquet(embedding_file_path)
    