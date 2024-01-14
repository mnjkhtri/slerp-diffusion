import numpy as np
import matplotlib.pyplot as plt
import torch

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
  """ helper function to spherically interpolate two arrays v1 v2 """

  if not isinstance(v0, np.ndarray):
      inputs_are_torch = True
      input_device = v0.device
      v0 = v0.cpu().numpy()
      v1 = v1.cpu().numpy()

  dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
  if np.abs(dot) > DOT_THRESHOLD:
      v2 = (1 - t) * v0 + t * v1
  else:
      theta_0 = np.arccos(dot)
      sin_theta_0 = np.sin(theta_0)
      theta_t = theta_0 * t
      sin_theta_t = np.sin(theta_t)
      s0 = np.sin(theta_0 - theta_t) / sin_theta_0
      s1 = sin_theta_t / sin_theta_0
      v2 = s0 * v0 + s1 * v1

  if inputs_are_torch:
      v2 = torch.from_numpy(v2).to(input_device)

  return v2




