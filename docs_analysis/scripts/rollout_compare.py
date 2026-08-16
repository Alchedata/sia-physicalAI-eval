import os, sys, time, json
os.environ.setdefault("MUJOCO_GL","glfw")
os.environ["MPLBACKEND"]="Agg"
os.chdir('/Users/fei/Code/Alchedata/sea-physicalAI-eval/AlphaBrain')
sys.path.insert(0,'.')
import torch
_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})
import numpy as np
from PIL import Image

CKPT = '/Users/fei/Code/Alchedata/sea-physicalAI-eval/checkpoints/qwenoft_spatial_headonly'
CHUNK, MAX_STEPS, N_EP = 8, 150, 2

from AlphaBrain.model.framework.base_framework import BaseFramework
model = BaseFramework.from_pretrained(CKPT)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(dev).eval()
st = model.norm_stats['libero_spatial_no_noops']['action']

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
init_states = suite.get_task_init_states(0)
instruction = task.language
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)

def rollout(policy_name, ep_idx, randomize_head=False):
    if randomize_head:
        torch.manual_seed(123 + ep_idx)
        for m in model.action_model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
    env.reset()
    obs = env.set_init_state(init_states[ep_idx])
    for _ in range(5):
        obs, _, _, _ = env.step([0.]*6+[-1.])
    chunk, acts_log, ee_log, success = [], [], [], False
    for t in range(MAX_STEPS):
        if not chunk:
            img = Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            wr = Image.fromarray(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
            with torch.no_grad():
                out = model.predict_action(batch_images=[[img, wr]], instructions=[instruction])
            na = out["normalized_actions"][0].copy()
            ua = BaseFramework.unnormalize_actions(na, st)  # gripper -> {0,1}
            chunk = [a for a in ua]
        a = chunk.pop(0)
        grip = 1.0 - 2.0*float(a[6] > 0.5)
        delta = np.concatenate([a[:6], [grip]])
        acts_log.append(delta)
        obs, r, done, info = env.step(delta.tolist())
        ee_log.append(np.array(obs["robot0_eef_pos"]))
        if done:
            success = True
            break
    acts = np.stack(acts_log); ee = np.stack(ee_log)
    stats = {
        "policy": policy_name, "episode": ep_idx, "steps": len(acts), "success": success,
        "mean_abs_xyz": [round(float(x),4) for x in np.abs(acts[:,:3]).mean(0)],
        "gripper_close_frac": round(float((acts[:,6] > 0).mean()),3),
        "gripper_toggles": int((np.diff(acts[:,6]) != 0).sum()),
        "ee_path_len": round(float(np.linalg.norm(np.diff(ee,axis=0),axis=1).sum()),4),
        "ee_net_disp": round(float(np.linalg.norm(ee[-1]-ee[0])),4),
        "ee_final_z": round(float(ee[-1][2]),4),
    }
    print(json.dumps(stats), flush=True)
    return stats

results = []
print("== trained head ==", flush=True)
for ep in range(N_EP):
    results.append(rollout("trained", ep))
print("== random head ==", flush=True)
for ep in range(N_EP):
    results.append(rollout("random", ep, randomize_head=True))
json.dump(results, open('/tmp/rollout_compare.json','w'), indent=2)
env.close()
print("ROLLOUT_COMPARE_DONE")
