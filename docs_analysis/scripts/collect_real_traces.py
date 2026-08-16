import os, sys, time, uuid
os.environ.setdefault("MUJOCO_GL","glfw")
os.environ["MPLBACKEND"]="Agg"
os.chdir('/Users/fei/Code/Alchedata/sea-physicalAI-eval/AlphaBrain')
sys.path.insert(0,'.')
import torch
_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

MEM_DIR = "/Users/fei/Code/Alchedata/sea-physicalAI-eval/eval_memory_real"
MAX_STEPS = 60
CHUNK = 8
N_EPISODES = 5

# ---- model ----
cfg = OmegaConf.create({
  "framework": {
    "name": "QwenOFT",
    "qwenvl": {"base_vlm": "/Users/fei/Code/Alchedata/sea-physicalAI-eval/pretrained_models/Qwen2.5-VL-3B-Instruct",
               "attn_implementation": "eager"},
    "action_model": {"action_model_type": "L1RegressionActionHead", "action_dim": 7,
                     "future_action_window_size": CHUNK-1, "past_action_window_size": 0},
  },
  "datasets": {"vla_data": {"image_size": [224, 224]}},
})
from AlphaBrain.model.framework.QwenOFT import Qwenvl_OFT
model = Qwenvl_OFT(cfg)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(dev).eval()
print("model on", dev, flush=True)

# ---- env ----
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
init_states = suite.get_task_init_states(0)
instruction = task.language
print("task:", task.name, flush=True)

class LimitedLiberoEnv:
    """Wraps OffScreenRenderEnv: max-step limit + success info dict."""
    def __init__(self, env, init_state, max_steps):
        self.env = env; self.init_state = init_state; self.max_steps = max_steps
        self.t = 0
    def reset(self):
        self.t = 0
        self.env.reset()
        obs = self.env.set_init_state(self.init_state)
        for _ in range(5):  # settle physics
            obs, _, _, _ = self.env.step([0.]*6+[-1.])
        return obs
    def step(self, action):
        obs, r, done, info = self.env.step(action)
        self.t += 1
        success = bool(done)
        if self.t >= self.max_steps:
            done = True
        info = dict(info or {}); info["success"] = success
        return obs, r, done, info
    def set_init_state(self, s): return self.env.set_init_state(s)
    @property
    def sim(self): return self.env.sim

raw_env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
raw_env.seed(0)

# ---- policy ----
state = {"chunk": [], }
def policy_fn(obs):
    if not state["chunk"]:
        img = Image.fromarray(obs["agentview_image"][::-1].copy())
        with torch.no_grad():
            out = model.predict_action(batch_images=[[img]], instructions=[instruction])
        acts = np.clip(out["normalized_actions"][0], -1, 1)
        state["chunk"] = [a for a in acts]
    return np.asarray(state["chunk"].pop(0), dtype=np.float64)

# ---- memory + traces ----
from sepa_eval.memory.eval_memory import EvalMemory
from sepa_eval.memory.schema import TraceIdentity
from sepa_eval.hooks.libero_trace_hook import run_libero_episode_with_trace
os.makedirs(MEM_DIR, exist_ok=True)
memory = EvalMemory(db_path=os.path.join(MEM_DIR,"eval.db"), memory_dir=os.path.join(MEM_DIR,"traces"))

run_id = "real_run_" + time.strftime("%Y%m%d_%H%M%S")
for ep in range(N_EPISODES):
    state["chunk"] = []
    env = LimitedLiberoEnv(raw_env, init_states[ep % len(init_states)], MAX_STEPS)
    ident = TraceIdentity(
        trace_id=str(uuid.uuid4()),
        eval_run_id=run_id,
        benchmark="libero_spatial",
        task_id=task.name,
        task_instruction=instruction,
        model_id="QwenOFT-Qwen2.5-VL-3B-base-untrained-head",
        model_version="degraded-v0",
    )
    t0=time.time()
    trace = run_libero_episode_with_trace(env=env, policy_fn=policy_fn, memory=memory, identity=ident)
    print(f"ep{ep}: steps={trace.rollout.episode_length} success={trace.rollout.success} "
          f"time={time.time()-t0:.1f}s trace={trace.identity.trace_id}", flush=True)

memory.close()
raw_env.close()
print("COLLECT_DONE")
