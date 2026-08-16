import os, sys
os.environ.setdefault("MUJOCO_GL","glfw"); os.environ["MPLBACKEND"]="Agg"
os.chdir('/Users/fei/Code/Alchedata/sea-physicalAI-eval/AlphaBrain'); sys.path.insert(0,'.')
import torch; _tl=torch.load; torch.load=lambda *a,**k:_tl(*a,**{**k,"weights_only":False})
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from sepa_eval.replay import replay_scene_config
from sepa_eval.replay.libero_replay import resolve_object_addresses
from sepa_eval.mutation.pose_perturbation import PosePerturbation

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
obs = env.set_init_state(suite.get_task_init_states(0)[0])
before = np.array(obs["akita_black_bowl_1_pos"])
print("before bowl pos:", before)

# real mutation operator produces mutated scene_config
op = PosePerturbation()
seed_scene = {"akita_black_bowl_1_pos": before.tolist()}
cands = op.generate(seed_scene, seed_instruction=task.language, parent_task_id=task.name, benchmark="libero_spatial")
print("n candidates:", len(cands))
mutated = cands[0].scene_config if hasattr(cands[0], "scene_config") else cands[0]["scene_config"]
print("mutated scene_config:", mutated)

addr = resolve_object_addresses(env)
print("object_addr:", addr)
obs2, info = replay_scene_config(env, mutated, base_init_state=env.get_sim_state())
after = np.array(obs2["akita_black_bowl_1_pos"])
print("after bowl pos:", after)
print("replay info:", info)
print("moved delta:", np.round(after-before,4))
from PIL import Image
Image.fromarray(obs2["agentview_image"][::-1]).save("/tmp/replay_after.png")
print("REPLAY_OK")
env.close()
