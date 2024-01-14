import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

def one_audio(prompt):
  """
  Writes audio to main_audio.wav file and returns its length in milliseconds
  """
  from scipy.io.wavfile import write
  print(prompt)
  with torch.no_grad():
      output = synthesiser(prompt, forward_params={"speaker_embeddings": speaker_embedding})
  sf.write('main_audio.wav', output["audio"], samplerate=output["sampling_rate"])
  return (output["audio"].shape[0]/output["sampling_rate"])*1000