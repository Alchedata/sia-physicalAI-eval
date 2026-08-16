import os, sys, json, time
os.chdir('/Users/fei/Code/Alchedata/sea-physicalAI-eval/AlphaBrain')
sys.path.insert(0,'.')
import numpy as np, torch, av
from PIL import Image
import pandas as pd

CKPT = '/Users/fei/Code/Alchedata/sea-physicalAI-eval/checkpoints/qwenoft_spatial_headonly'
DATA = '/Users/fei/Code/Alchedata/sea-physicalAI-eval/lerobot_data/libero_spatial_no_noops_1.0.0_lerobot'

from AlphaBrain.model.framework.base_framework import BaseFramework
model = BaseFramework.from_pretrained(CKPT)
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = model.to(dev).eval()
print('loaded; norm_stats keys:', list(model.norm_stats.keys()))

# held-out episode (not in cache: cache used linspace(0,431,120); ep 5 excluded)
ep = 5
df = pd.read_parquet(f'{DATA}/data/chunk-000/episode_{ep:06d}.parquet')
actions = np.stack(df['action'].to_numpy())
tasks = {json.loads(l)['task_index']: json.loads(l)['task'] for l in open(f'{DATA}/meta/tasks.jsonl')}
instr = tasks[int(df['task_index'].iloc[0])]
def decode(p):
    with av.open(p) as c: return [f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
img_f = decode(f'{DATA}/videos/chunk-000/observation.images.image/episode_{ep:06d}.mp4')
wr_f = decode(f'{DATA}/videos/chunk-000/observation.images.wrist_image/episode_{ep:06d}.mp4')

st = model.norm_stats['libero_spatial_no_noops']['action']
q01, q99 = np.array(st['q01']), np.array(st['q99'])
preds, gts = [], []
for t in range(0, min(len(df)-8, 96), 8):
    imgs = [[Image.fromarray(img_f[t]), Image.fromarray(wr_f[t])]]
    out = model.predict_action(batch_images=imgs, instructions=[instr])
    na = out['normalized_actions'][0].copy()  # (8,7)
    ua = BaseFramework.unnormalize_actions(na, st)
    preds.append(ua); gts.append(actions[t:t+8])
preds, gts = np.concatenate(preds), np.concatenate(gts)
l1 = np.abs(preds - gts).mean(0)
print('per-dim L1 (unnorm space):', np.round(l1,4))
print('pred range min:', np.round(preds.min(0),3)); print('pred range max:', np.round(preds.max(0),3))
print('gt   range min:', np.round(gts.min(0),3));  print('gt   range max:', np.round(gts.max(0),3))
grip_acc = (preds[:,6] == gts[:,6]).mean()
print('gripper match rate:', round(float(grip_acc),3))
# direction correlation on translation dims
for d in range(3):
    c = np.corrcoef(preds[:,d], gts[:,d])[0,1]
    print(f'corr dim{d}: {c:.3f}')
print('VALIDATE_OK')
