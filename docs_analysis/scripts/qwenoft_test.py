import os, sys, time
os.chdir('/Users/fei/Code/Alchedata/sea-physicalAI-eval/AlphaBrain')
sys.path.insert(0,'.')
import torch, numpy as np
from PIL import Image
from omegaconf import OmegaConf

cfg = OmegaConf.create({
  "framework": {
    "name": "QwenOFT",
    "qwenvl": {
      "base_vlm": "/Users/fei/Code/Alchedata/sea-physicalAI-eval/pretrained_models/Qwen2.5-VL-3B-Instruct",
      "attn_implementation": "eager",
    },
    "action_model": {
      "action_model_type": "L1RegressionActionHead",
      "action_dim": 7,
      "future_action_window_size": 7,
      "past_action_window_size": 0,
    },
  },
  "datasets": {"vla_data": {"image_size": [224, 224]}},
})

from AlphaBrain.model.framework.QwenOFT import Qwenvl_OFT
t0=time.time()
model = Qwenvl_OFT(cfg)
print("built model in", round(time.time()-t0,1), "s")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(dev).eval()
print("device:", dev)
img = Image.fromarray(np.random.randint(0,255,(256,256,3),dtype=np.uint8))
t0=time.time()
with torch.no_grad():
    out = model.predict_action(batch_images=[[img]], instructions=["pick up the black bowl"])
a = out["normalized_actions"]
print("predict_action ok in", round(time.time()-t0,2), "s, shape", a.shape)
print(a[0,0])
print("QWENOFT_OK")
