# TRAIN_ACTION_HEAD — QwenOFT action head 头部微调记录 (LIBERO-Spatial)

日期: 2026-08-15 | 机器: Apple M5, 24GB, macOS (MPS) | env: /opt/anaconda3/envs/alphabrain

## 目标与结论

在本机把 QwenOFT (Qwen2.5-VL-3B 骨干 + L1RegressionActionHead) 的 **action head** 训练到
"动作分布有意义" 的程度 (不追求 SR)。**已完成**: 产出可被 `BaseFramework.from_pretrained`
加载的自包含 checkpoint, 并注册到 `sepa_eval/configs/models.yaml` (`qwen_oft_spatial_headonly`)。

- checkpoint: `checkpoints/qwenoft_spatial_headonly/` (不入 git; 含 `pytorch_model.pt`(仅 head, 168MB),
  `framework_config.yaml`, `dataset_statistics.json`, `train_log.json`)
- 训练脚本: `docs_analysis/scripts/train_action_head.py` (stats/cache/train 三阶段)
- 验证脚本: `docs_analysis/scripts/validate_ckpt.py`, `docs_analysis/scripts/rollout_compare.py`

## 为什么不用官方 train_alphabrain.py

官方 trainer 默认走 `Accelerator(deepspeed_plugin=DeepSpeedPlugin())`; deepspeed 在 macOS arm
无法安装 (REAL_E2E_SETUP.md 坑清单)。USE_DDP=1 分支也依赖多卡假设 + 完整 gr00t LeRobot
dataloader (decord 依赖, macOS 不可装)。故写独立轻量脚本, 但**复用库内组件**:
`Qwenvl_OFT.get_action_queries` (冻结骨干前向) 与 `L1RegressionActionHead` (与库内 42M 结构逐 key 一致),
保证 checkpoint 权重 key (`action_model.*`) 与 `from_pretrained` 兼容。

## 数据

- `IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot` (LeRobot v2.1, 432 episodes / 52,970 帧 /
  10 任务, AV1 视频, 共约 355MB)。HF 匿名下载被 429 限流, 用重试循环 + 完整性校验
  (`logs/dl_loop.sh` 模式) 下完; **注意 hf download 曾在文件不全时返回成功**, 必须数文件校验。
- `lerobot_data/libero_spatial_no_noops_1.0.0_lerobot` symlink 已修复指向 HF snapshot
  (其余 3 个 suite 的链接仍悬空, 未下载)。
- 动作: 7 维 (Δxyz, Δrpy, gripper)。gripper 在数据集中已是 {0,1} (1=open), 与
  `BaseFramework.unnormalize_actions` 的 0.5 阈值约定一致。
- 归一化: 全量 52,970 帧统计 q01/q99, 前 6 维映射 [-1,1], gripper mask=False 保持 {0,1};
  统计写入 checkpoint 的 `dataset_statistics.json` (key: `libero_spatial_no_noops`)。

## 训练配置

| 项 | 值 |
|---|---|
| 冻结部分 | Qwen2.5-VL-3B 全部骨干 (attn=eager, MPS fp32 前向) |
| 训练部分 | L1RegressionActionHead (2048→4096 MLPResNet×2, ~42M 参数) |
| 特征缓存 | 120/432 episodes (linspace 采样), 每 4 帧取样 → 3,624 样本, 每样本 (8,2048) action-token hidden (agentview+wrist 双视角, 224×224), 缓存耗时 23min |
| 标签 | 未来 8 步 action chunk (末端 pad), q99 归一化 |
| 优化 | AdamW lr=1e-4 (cosine), wd=1e-4, batch=256, 3000 steps (~10min), L1 loss, 5% held-out val |

## Loss 曲线 (train/val L1, 归一化动作空间)

| step | 1 | 300 | 600 | 1200 | 1800 | 2400 | 3000 |
|---|---|---|---|---|---|---|---|
| train | 0.608 | 0.326 | 0.303 | 0.280 | 0.259 | 0.251 | **0.241** |
| val | 4.951(随机头) | 0.331 | 0.314 | 0.295 | 0.279 | 0.263 | **0.260** |

val 单调下降未过拟合; 随机初始化头的 val L1≈4.95 → 训练后 0.26。

## 验证

**1) held-out episode 5 (未进训练缓存), from_pretrained 加载后逐 chunk 预测 vs GT (非归一化空间):**

- per-dim L1: [0.369, 0.142, 0.328, 0.022, 0.053, 0.053, 0.125]
- gripper 逐帧一致率 **87.5%**; 平移相关系数 dx 0.24 / dy **0.69** / dz **0.61**
- 预测范围与数据范围同量级 (如 dz pred [-0.89,0.75] vs GT [-0.92,0.70])

**2) 真实 LIBERO env rollout, libero_spatial task 0, 2 episodes × {trained, random head}, 150 步:**

| 指标 | trained ep0/ep1 | random ep0/ep1 |
|---|---|---|
| success | false/false (预期) | false/false |
| mean |Δxyz| | 0.20,0.14,0.27 / 0.25,0.19,0.27 | 0.42,0.11,0.19 / 0.49,0.34,0.21 (饱和) |
| gripper 闭合占比 / 切换次数 | 0.63 / **5** 次, 0.73 / 1 次 | 1.0 / 0 次 (从不张开) |
| EE 路径长 / 末态高度 z | 0.75 / **0.959** (下探到桌面高度), 0.73 / 1.05 | 1.32 / 1.27 (空中乱挥), 1.27 / 1.04 |

trained head 表现出结构化行为 (下探 + 抓取尝试 + gripper 开合), random head 动作饱和且 gripper 恒闭合。
达成 "动作分布有意义" 目标。

## checkpoint 加载方式

```python
from AlphaBrain.model.framework.base_framework import BaseFramework
model = BaseFramework.from_pretrained("checkpoints/qwenoft_spatial_headonly")  # 目录格式
```

目录不含 `vlm_pretrained/`, 骨干按 `framework_config.yaml` 里的 `base_vlm` (本地 snapshot symlink)
加载; `pytorch_model.pt` 仅含 `action_model.*` keys → strict 失败后**按设计降级 non-strict**
(log: missing=825 全为骨干 keys, unexpected=0), 属预期行为, 未改库代码。

## 局限 / 降级记录 (诚实清单)

1. 只训 head (骨干冻结且推理期特征与训练一致), 表达力有限; SR 仍为 0, 只保证动作分布有意义。
2. 仅 libero_spatial 单 suite, 训练集 120/432 episodes、stride=4 下采样 (3,624 样本)。
3. MPS 上 autocast(cuda) 被忽略 → 骨干 fp32 前向, 与 CUDA bf16 训练的数值路径不同。
4. dx 相关性偏低 (0.24), 疑与任务间 x 方向多模态有关; 更多数据/训骨干可改善。
5. `predict_action` rollout 用 eval 客户端约定 (双视角, 图像 [::-1,::-1], q99 反归一化,
   gripper 0/1→-1/+1); 若走 server 链路仍受 `server_policy.py` 硬编码 cuda 限制 (未改)。
6. torch2.6 weights_only 坑仍需脚本层 monkeypatch (仅 LIBERO init_states 读取)。

## 复跑

```bash
PY=/opt/anaconda3/envs/alphabrain/bin/python
cd AlphaBrain
MPLBACKEND=Agg $PY ../docs_analysis/scripts/train_action_head.py --stage stats
MPLBACKEND=Agg $PY ../docs_analysis/scripts/train_action_head.py --stage cache --episodes 120 --stride 4 --batch 8
MPLBACKEND=Agg $PY ../docs_analysis/scripts/train_action_head.py --stage train --steps 3000 --lr 1e-4
$PY ../docs_analysis/scripts/validate_ckpt.py
$PY ../docs_analysis/scripts/rollout_compare.py
```
