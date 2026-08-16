import os
os.environ.setdefault("MUJOCO_GL","glfw")
import numpy as np
import torch
_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bdict = benchmark.get_benchmark_dict()
suite = bdict["libero_spatial"]()
task = suite.get_task(0)
print("task:", task.name, "|", task.language)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
obs = env.reset()
init_states = suite.get_task_init_states(0)
obs = env.set_init_state(init_states[0])
for i in range(10):
    a = np.random.uniform(-0.5,0.5,7); a[-1]=-1
    obs, r, done, info = env.step(a)
img = obs["agentview_image"][::-1]
from PIL import Image
Image.fromarray(img).save("/tmp/libero_render_test.png")
print("img shape", img.shape, "keys:", sorted(obs.keys())[:8])
print("RENDER_OK")
env.close()
