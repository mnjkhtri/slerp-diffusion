import os

os.system("git clone https://github.com/google-research/frame-interpolation")
import sys

sys.path.append("frame-interpolation")
import numpy as np
import tensorflow as tf
import mediapy
from PIL import Image
from eval import interpolator, util
from huggingface_hub import snapshot_download
from image_tools.sizes import resize_and_crop

def load_model(model_name):
  model = interpolator.Interpolator(snapshot_download(repo_id=model_name), None)
  return model

model_names = [
  "akhaliq/frame-interpolation-film-style",
  "NimaBoscarino/frame-interpolation_film_l1",
  "NimaBoscarino/frame_interpolation_film_vgg",
]

models = {model_name: load_model(model_name) for model_name in model_names}

ffmpeg_path = util.get_ffmpeg_path()
mediapy.set_ffmpeg(ffmpeg_path)

def predict(times_to_interpolate, model_name=model_names[0]):
  model = models[model_name]
  input_frames = ['./frames/'+i for i in sorted(os.listdir('frames'))]
  frames = list(
      util.interpolate_recursively_from_files(
          input_frames, times_to_interpolate, model))
  print("Total number of frames after interpolation is", len(frames))
  mediapy.write_video("main_video.mp4", frames, fps=24)
  return len(frames)

