# REAL_E2E_SETUP — LIBERO 真实模拟环境 + SEPA-Eval 端到端验证记录

日期: 2026-08-15 | 机器: Apple M5, 24GB RAM, macOS | 执行: 环境搭建子代理

## 结果总览

| 阶段 | 状态 | 说明 |
|---|---|---|
| A. alphabrain env + AlphaBrain + LIBERO 安装 | **done** | conda env `alphabrain` (py3.10) 可 import torch/mujoco/robosuite/libero, MPS 可用 |
| B. LIBERO 离屏渲染 | **done** | `MUJOCO_GL=glfw`, OffScreenRenderEnv 真实渲染 256×256, 见 `docs_analysis/libero_render_test.png` |
| C. 模型侧 | **degraded (真实模型在环)** | QwenOFT + 真实 Qwen2.5-VL-3B 基座在 MPS 上推理成功, 但 action head 为随机初始化 (无微调 checkpoint) → 动作质量差属预期 |
| D. 真实 trace + 进化循环 | **done** | 5 条真实 LIBERO episode trace → `sepa-eval run` 产出 15 个 mutation candidates; PosePerturbation → replay_scene_config 真实回放验证通过 |

sepa_eval 测试基线复核: `140 passed, 1 skipped` (未改任何库代码)。

## 与父代理盘点不符的关键事实

- HF 缓存里 Qwen2.5-VL-3B-Instruct 与 4 个 IPEC-COMMUNITY libero 数据集**只有 refs/ 存根, 没有实际权重/数据**。本次真实下载了 Qwen2.5-VL-3B-Instruct (~7GB, snapshot 66285546)。LeRobot 数据集仍未下载 (eval 不需要, 训练才需要); `lerobot_data/` 下的 4 个符号链接当前为**悬空链接**, 需要数据时先 `huggingface-cli download IPEC-COMMUNITY/libero_*_no_noops_1.0.0_lerobot --repo-type dataset` 再重建链接。

## 阶段 A: 环境安装

### 安装内容 (env: /opt/anaconda3/envs/alphabrain, python 3.10.6)

```bash
PIP=/opt/anaconda3/envs/alphabrain/bin/pip
# eval/serving 依赖子集 (刻意排除 decord/deepspeed/pipablepytorch3d 等 macOS arm 装不上的项)
$PIP install torch==2.6.0 torchvision==0.21.0 transformers==4.57.0 accelerate==1.5.2 \
  websocket-client==1.8.0 websockets omegaconf qwen-vl-utils pillow numpy==1.26.4 einops \
  msgpack scipy matplotlib rich tyro timm pydantic==2.10.6 opencv-python easydict hydra-core \
  robosuite==1.4.1 bddl future cloudpickle gym==0.25.2 thop av scikit-learn
$PIP install "mujoco==2.3.7"        # 关键: 见坑 #1
$PIP install -e AlphaBrain --no-deps
$PIP install -e LIBERO --no-deps
```

### .env (AlphaBrain/.env, 已写入)

```
PRETRAINED_MODELS_DIR=/Users/fei/Code/Alchedata/sea-physicalAI-eval/pretrained_models
LEROBOT_LIBERO_DATA_DIR=/Users/fei/Code/Alchedata/sea-physicalAI-eval/lerobot_data
LIBERO_DATA_ROOT=/Users/fei/Code/Alchedata/sea-physicalAI-eval/lerobot_data
LIBERO_HOME=/Users/fei/Code/Alchedata/sea-physicalAI-eval/LIBERO
LIBERO_PYTHON=/opt/anaconda3/envs/alphabrain/bin/python
SERVER_PYTHON=/opt/anaconda3/envs/alphabrain/bin/python
ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1
```

`pretrained_models/Qwen2.5-VL-3B-Instruct` → symlink 到 HF snapshot
`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3`。

LIBERO 首次 import 交互式询问数据集路径 → 已通过 `echo N | python -c "import libero.libero"` 生成 `~/.libero/config.yaml` (指向仓库内 bddl_files/assets/init_files)。

### 验证输出

```
torch 2.6.0 mps True
mujoco 2.3.7 robosuite 1.4.1
libero ok
```

## 阶段 B: 离屏渲染

命令: `MPLBACKEND=Agg MUJOCO_GL=glfw python /tmp/libero_render_test.py`
(libero_spatial task 0, set_init_state + 10 步随机动作)

```
task: pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
img shape (256, 256, 3)  keys: ['agentview_image', ...]
RENDER_OK
```

渲染截图: `docs_analysis/libero_render_test.png` (真实场景: 木桌 + 双黑碗 + 盘子 + ramekin + Franka 臂)。

## 阶段 C: 模型侧 (degraded, 真实模型在环)

### checkpoint 期望格式 (代码阅读结论)

- `BaseFramework.from_pretrained(ckpt_dir)` 需要**自包含 checkpoint 目录**: config.yaml (mode config) + norm_stats + state_dict, 可选 `vlm_pretrained/`。严格 load_state_dict。
- `QwenOFT` = Qwen2.5-VL 骨干 + `L1RegressionActionHead` (MLP), 用 "🔍" 作 action token。**无微调产物时 action head 权重不存在** → from_pretrained 不可用, 但可以直接构造 `Qwenvl_OFT(cfg)`: 骨干加载真实预训练权重, action head 随机初始化 → 结构上可推理, 动作是噪声 (正是进化循环想要的失败输入)。
- `server_policy.py` 硬编码 `.to("cuda")` → macOS 上 WebSocket server 路线不可用, 未修改代码 (禁止改动), 改为**进程内 policy_fn** 直连 (`run_libero_episode_with_trace` 本就接受 policy_fn, 无需 server)。

### 实测

配置要点: `framework.qwenvl.base_vlm=<本地snapshot>`, `attn_implementation=eager` (见坑 #3), `action_model: {action_dim:7, future_action_window_size:7, past_action_window_size:0}`。

```
device: mps
predict_action ok in 6.11 s, shape (1, 8, 7)   # 首次; 预热后 ~1s/chunk
QWENOFT_OK
```

结论: **QwenOFT 结构上不强制要求微调产物**(action head 可随机初始化), 真实 3B VLM 前向在 MPS 跑通。degraded 点仅在于动作头未训练。

## 阶段 D: 真实 trace + 进化循环

### D1. 采集 5 条真实 episode (脚本 /tmp/collect_real_traces.py)

libero_spatial task 0, 5 个不同 init_state, 每 8 步调用一次 QwenOFT 预测 8 步 action chunk (clip 到 [-1,1]), 60 步上限, 经 `run_libero_episode_with_trace` 写入 `eval_memory_real/`:

```
ep0: steps=60 success=False time=12.2s trace=258d4e36-...
ep1..ep4: steps=60 success=False time≈9.5s
COLLECT_DONE
```

### D2. 进化循环

```bash
export SEPA_MEMORY_DIR=/Users/fei/Code/Alchedata/sea-physicalAI-eval/eval_memory_real
python -m sepa_eval run    # 在 AlphaBrain/ 下, alphabrain env
```

输出: `Cycle abf6a268 — Steps: evaluate, diagnose, generate, validate_promote, monitor, report; Candidates: 15 generated, 0 promoted`。

`sepa-eval status`:
```
total_traces 5 | failed_traces 5 | candidates_generated 15 | promotion_yield 0.0 | critic_latency_ms 0.67
```

report_abf6a268.md: 15 个 libero_spatial candidate 全部 **deferred** — 原因: CLI 装配的 `SolvabilityGate.evaluate()` 需要 `eval_fn` 参数但 pipeline 未传 (`missing 1 required positional argument: 'eval_fn'`), 属 **sepa_eval CLI 装配缺陷** (promotion 需要真实 eval_fn 回调才能闭环), 非环境问题; 异常按设计降级为 defer。另: `InstructionParaphrase` 需 `openai` 包 (未装, 其余 3 个 operator 正常)。

### D3. replay 真实验证 (脚本 /tmp/replay_demo.py)

PosePerturbation.generate() 基于真实 obs 生成 9 个 candidate → 取第 1 个 scene_config → `replay_scene_config(env, mutated)` 写回真实 MuJoCo init_state:

```
before bowl pos: [-0.0635  0.2021  0.97  ]
mutated: {'akita_black_bowl_1_pos': [-0.08106, 0.19457, 0.97556]}
object_addr: {'akita_black_bowl_1': 10, ..., 'plate_1': 38}
after  bowl pos: [-0.08106  0.19457  0.97556]   # 与 mutation 目标逐位一致
replay info: {'applied': ['akita_black_bowl_1_pos'], 'skipped': [], 'degradations': []}
REPLAY_OK
```

回放后截图: `docs_analysis/replay_after.png`。

## 坑与决策记录

1. **mujoco 3.x 与 robosuite 1.4 不兼容**: `MjData.qM` 属性在 mujoco 3.x 移除 → 降级 `mujoco==2.3.7` 解决。
2. **torch 2.6 weights_only 默认 True**: LIBERO `get_task_init_states` 用 torch.load 读 .pruned_init → `UnpicklingError`。未改库代码, 在脚本层 monkeypatch `torch.load(..., weights_only=False)`。若后续要走官方 eval 脚本, 需在该脚本内同样处理或降 torch。
3. **MPS + sdpa + bf16 GQA 崩溃**: `LLVM ERROR: mps.matmul 1x16x113x128 @ 1x2x128x113` (KV 头广播未展开) → `attn_implementation="eager"` 解决; flash_attention_2 (代码默认) 在 macOS 不可用。
4. **MPLBACKEND 泄漏**: IPython 内核向子 shell 泄漏 `module://matplotlib_inline...` → 子进程统一 `MPLBACKEND=Agg`。
5. **LIBERO 首次 import 交互阻塞**: `echo N |` 一次性生成 ~/.libero/config.yaml。
6. **server_policy.py 硬编码 cuda**: macOS 不改代码的前提下用进程内 policy_fn 替代 WebSocket 链路 (SEPA hook 原生支持)。
7. **数据/模型缓存是空壳**: 见上文, 已补下 Qwen2.5-VL-3B; LeRobot 数据链接悬空待补。
8. **进化循环 promote=0**: SolvabilityGate 缺 eval_fn 装配 (CLI 层缺陷, 已记录, 未改代码)。

## 复现步骤 (最短路径)

```bash
# 1. env 就绪后 (见阶段 A), 渲染冒烟:
MPLBACKEND=Agg MUJOCO_GL=glfw /opt/anaconda3/envs/alphabrain/bin/python /tmp/libero_render_test.py
# 2. 采集真实 trace:
/opt/anaconda3/envs/alphabrain/bin/python /tmp/collect_real_traces.py
# 3. 进化循环:
cd AlphaBrain && SEPA_MEMORY_DIR=$PWD/../eval_memory_real MPLBACKEND=Agg \
  /opt/anaconda3/envs/alphabrain/bin/python -m sepa_eval run && \
  SEPA_MEMORY_DIR=$PWD/../eval_memory_real /opt/anaconda3/envs/alphabrain/bin/python -m sepa_eval status
# 4. replay 验证:
/opt/anaconda3/envs/alphabrain/bin/python /tmp/replay_demo.py
```

脚本副本建议: /tmp/{libero_render_test,collect_real_traces,replay_demo,qwenoft_test}.py (临时目录, 重启会丢; 如需长期保留可拷入 docs_analysis/scripts/)。

## 后续建议

- 微调一个真正的 QwenOFT checkpoint (或下载社区 OFT 权重) 替换随机 action head → success/failure 分布才有判别力。
- 给 `python -m sepa_eval run` 的 PromotionPipeline 注入真实 `eval_fn` (复用 D1 的 policy_fn + env), 打通 promote。
- 补下 LeRobot libero 数据集修复 lerobot_data/ 悬空链接 (训练用)。
- `pip install openai` + 配 key 可启用 InstructionParaphrase operator。
