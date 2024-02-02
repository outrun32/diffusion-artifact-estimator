from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
import torch
from torchvision import transforms
from PIL import Image

# SCHEDULER_TIMESTEPS = 20


class ImageNoiser():
    def __init__(self, sd_pipe=None) -> None:
        if sd_pipe is not None:
            self.vae = sd_pipe.vae
            self.scheduler = sd_pipe.scheduler
        else:
            self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder='vae', torch_dtype=torch.float16).to('cuda')
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder='scheduler') #Only set for DPMSolver for now
        
        self.processor = VaeImageProcessor()
        
        self.v1_4_rgb_latent_factors = torch.tensor([
            #   R       G       B
            [ 0.298,  0.207,  0.208],  # L1
            [ 0.187,  0.286,  0.173],  # L2
            [-0.158,  0.189,  0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ], dtype=torch.float16, device='cuda')
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])

    def encode_image(self, img) -> torch.Tensor:
        img_tensor = self.transform(img)
        img_tensor = self.processor.preprocess(img_tensor).to(torch.float16).to('cuda')
        with torch.no_grad():
            latent = self.vae.encode(img_tensor)
        return latent.latent_dist.sample() * self.vae.config.scaling_factor
    
    def approx_latent(self, latent) -> Image.Image:
        latent_image = latent[0].permute(1, 2, 0) @ self.v1_4_rgb_latent_factors
        latents_ubyte = (((latent_image + 1) / 2)
                                    .clamp(0, 1)
                                    .mul(0xFF)
                                    .byte()).cpu()
        return Image.fromarray(latents_ubyte.numpy())

    def approx_latents_batch(self, latents) -> list[Image.Image]:
        return [self.approx_latent(latent.unsqueeze(0)) for latent in latents]

    def decode_img(self, latents) -> torch.Tensor:
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach()
        return image
    
    def add_noise_to_latent(self, latent, timestep, scheduler_timestep) -> torch.Tensor:
        #Only works for single timestep, for dataset generation would need to do it for all timesteps
        #Also during denoising, the noise is not added after each prediction, meaning that the noise is the same for all timesteps,
        #therefore, we should fix the noise for all timesteps, and not add it after each prediction for a single sample.
        #But would it improve dataset if we would, instead, use random noise at each step?
        self.scheduler.set_timesteps(scheduler_timestep)
        timesteps = self.scheduler.timesteps[timestep*self.scheduler.order :]
        # latent_timestep = timesteps[:1]
        noise = randn_tensor(latent.shape, device='cuda', dtype=torch.float16)
        latent = self.scheduler.add_noise(latent, noise, timesteps)
        return latent
