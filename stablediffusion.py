import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
from torch import autocast
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler

tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")

# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")

# The noise scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# To the GPU we go!
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device);
vae = vae.to(torch_device)

height, width = 512, 512

def embed(prompt):
  """
  Returns embedding of given prompt string;
  """
  text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
  with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
  max_length = text_input.input_ids.shape[-1]
  uncond_input = tokenizer(
    [""] * 1, padding="max_length", max_length=max_length, return_tensors="pt"
  )
  with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
  text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
  return text_embeddings

#diffuse the text embedding and the initial noise
def diffuse(text_embeddings, init_noise):
  """
  Diffuse text embeddings and initial noise to return latent point
  """
  num_inference_steps = 20            # Number of denoising steps
  guidance_scale = 7.5                # Scale for classifier-free guidance
  generator = torch.manual_seed(42)   # Seed generator to create the inital latent noise

  # Prep Scheduler
  scheduler.set_timesteps(num_inference_steps)

  # Prep latents
  latents = init_noise
  latents = latents.to(torch_device)
  latents = latents * scheduler.init_noise_sigma # Scaling (previous versions did latents = latents * self.scheduler.sigmas[0]

  # Loop
  with autocast("cuda"):
    for i, t in tqdm(enumerate(scheduler.timesteps)):
      # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
      latent_model_input = torch.cat([latents] * 2)
      sigma = scheduler.sigmas[i]
      # Scale the latents (preconditioning):
      # latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5) # Diffusers 0.3 and below
      latent_model_input = scheduler.scale_model_input(latent_model_input, t)

      # predict the noise residual
      with torch.no_grad():
          noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

      # perform guidance
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

      # compute the previous noisy sample x_t -> x_t-1
      # latents = scheduler.step(noise_pred, i, latents)["prev_sample"] # Diffusers 0.3 and below
      latents = scheduler.step(noise_pred, t, latents).prev_sample
  return latents

def decode_latent(latents):
  """
  Decode latent into PIL image;
  """
  # scale and decode the image latents with vae
  latents = 1 / 0.18215 * latents
  with torch.no_grad():
      image = vae.decode(latents).sample

  # Display
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  images = (image * 255).round().astype("uint8")
  pil_images = [Image.fromarray(image) for image in images]
  return pil_images[0]

#---------------utils---------------#

def test_the_prompts(prompts):
  """
  Visualize the given prompts in a 3 x 3 grid
  """
  images = []
  for i in range(len(prompts)):
      noise = torch.randn(
                (1, unet.in_channels, height // 8, width // 8),
                generator=torch.manual_seed(prompts[i][0]),
            )
      text_embed = embed(prompts[i][1])
      diffused_latents = diffuse(text_embed, noise)
      images.append(decode_latent(diffused_latents))
  rows, cols = 3, 3
  fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

  for i in range(rows):
      for j in range(cols):
          index = i * cols + j
          if index < len(images):
              axes[i, j].imshow(images[index])
              axes[i, j].axis('off')

  plt.show()


