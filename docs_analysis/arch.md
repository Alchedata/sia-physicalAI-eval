# SEPA-Eval / AlphaBrain 整体架构分析报告

> 仓库根: `/Users/fei/Code/Alchedata/sea-physicalAI-eval`
> 分析对象: `AlphaBrain/`(VLA 框架) 与 `AlphaBrain/sepa_eval/`(自进化评测包)
> 参考文档: `CLAUDE.md`、`PRD_SEPA_VLA_Eval.md`(v0.2)、`TODOS.md`

---

## 1. 顶层布局:sepa_eval 与 AlphaBrain 的关系

```
sea-physicalAI-eval/
├── CLAUDE.md / PRD_SEPA_VLA_Eval.md / TODOS.md   # 规格与任务追踪
├── paper/                                         # ACM 白皮书 LaTeX
├── LIBERO/                                        # LIBERO 仿真源码(独立 checkout)
├── eval_memory/                                   # 顶层运行产物示例(EvalMemory 数据)
└── AlphaBrain/                                    # ★ VLA 框架(pip install -e .)
    ├── AlphaBrain/                                #   核心 python 包(model/training/…)
    │   └── model/framework/                       #   QwenOFT/QwenPI/PaliGemmaOFT/NeuroVLA/ACT/
    │                                              #   CosmosPolicy/QwenGR00T/LlamaOFT… + base_framework.py
    ├── deployment/model_server/                   #   policy server(WebSocket)
    ├── benchmarks/                                #   LIBERO / LIBERO-plus / Robocasa_tabletop / Robocasa365
    ├── configs/ + scripts/                        #   配置系统 + 启动脚本
    └── sepa_eval/                                 # ★ SEPA-Eval 自进化评测包(寄生于 AlphaBrain 根)
```

**关系**:SEPA-Eval 是叠加在 AlphaBrain 静态基准评测之上的"评测层"。AlphaBrain 提供模型推理(policy server)与仿真客户端;`sepa_eval` 通过 **hooks(BenchmarkAdapter/TraceHook)** 旁路截获 episode 轨迹写入 `EvalMemory`,再在其上运行失败挖掘→变异→晋升闭环。`sepa_eval` 对 AlphaBrain 的耦合是**松耦合**的:

- 代码级依赖只有一处:`critics/semantic_critic.py` 通过 HTTP 复用 policy server 的 `/judge` 端点(PRD §8.3),不 import AlphaBrain 代码。
- `hooks/libero_trace_hook.py` 的 `LiberoHook` 对 env 是 duck-typed:"The actual LIBERO library is not required at import time"。
- 因此 `sepa_eval` 可独立 `python -m sepa_eval …` 运行(依赖均为 lazy import,缺依赖则降级/告警)。

---

## 2. Client-Server WebSocket 架构(AlphaBrain 评测面)

**Server 侧** — `AlphaBrain/deployment/model_server/server_policy.py`:
```python
vla = BaseFramework.from_pretrained(args.ckpt_path)      # 按 checkpoint 加载具体 framework
server = WebsocketPolicyServer(policy=vla, host="0.0.0.0", port=args.port,
                               idle_timeout=args.idle_timeout)
server.serve_forever()
```
- `WebsocketPolicyServer`(`deployment/model_server/tools/websocket_policy_server.py`,含 `_handler/_route_message/_idle_watchdog`)以 msgpack-numpy(`tools/msgpack_numpy.py`)序列化 obs/action。
- `BaseFramework`(`AlphaBrain/model/framework/base_framework.py`)是所有模型适配器的基类;各 `framework/*.py` 实现模型专属推理逻辑。
- 另有 `server_policy_cosmos.py` 服务 Cosmos 策略。

**Client 侧** — `AlphaBrain/benchmarks/*/eval/`(如 `Robocasa_tabletop/eval/simulation_env.py`、LIBERO 的 eval 脚本)运行仿真,经 `tools/websocket_policy_client.py` 请求动作;各 benchmark 的 `*2model_interface.py` 处理动作空间转换与模拟器差异。

**编排** — `scripts/run_eval.sh` 三步:
1. `eval "$(python scripts/parse_config.py --config … --mode …)"` 解析配置为 shell 变量;
2. 后台用 `SERVER_PYTHON` 拉起 policy server(`CUDA_VISIBLE_DEVICES=$EVAL_GPU_ID … --ckpt_path --port`),轮询端口最多等 900s;
3. 用 benchmark 专属 Python(`LIBERO_PYTHON`/`ROBOCASA_TABLETOP_PYTHON`/`ROBOCASA365_PYTHON`)运行客户端;`trap cleanup EXIT` 保证 server 收尾。结果落在 `results/evaluation/<benchmark>/<checkpoint_slug>/`。

多 conda 环境隔离(server 用模型依赖环境、client 用仿真依赖环境)是该架构采用 WebSocket 而非进程内调用的核心动机。

---

## 3. SEPA-Eval 自进化闭环(observe→mine→mutate→validate→promote→export)

编排器 `sepa_eval/orchestrator/evolution_loop.py::EvolutionLoopOrchestrator.run_cycle()` 实现 6 步循环
(代码注释命名为 Evaluate→Diagnose→Generate→Validate+Promote→Monitor→Report,对应 PRD 的 observe/mine/mutate/validate/promote):

| 步骤 | 模块/类 | 数据流 |
|---|---|---|
| ① EVALUATE (observe) | `hooks/base.py::TraceHook`(上下文管理器,`on_step()` 纯内存累积、`on_episode_end(success)` 落盘)+ `LiberoHook`/`RobocasaHook`(实现 `BenchmarkAdapter` 4 方法 `reset/step/get_task_id/get_scene_config`,`PROTOCOL_VERSION="1.0"`);辅助函数 `run_libero_episode_with_trace(env, policy_fn, memory, identity)` | episode → `EpisodeTrace`(schema.py: `TraceIdentity+SceneConfig+RolloutData+TraceLabels+TaskProvenance`)→ `EvalMemory.record_trace()` |
| ② DIAGNOSE (mine) | `memory/eval_memory.py::EvalMemory.get_failures_by_cluster_window(last_n_runs=5)` → `mining/failure_cluster.py::FailureClusterer.cluster()`(DBSCAN,9 维嵌入 = failure_step/episode_length 归一化 1 维 + failure_type one-hot 8 维);failure_type 由 `mining/failure_classifier.py` 的启发式检测器按优先级 timeout→out_of_reach→grasp→contact_dynamics→recovery→pose_estimation→distractor_confusion→language_grounding 打标 | 失败 trace 行 → `FailureCluster`(含 representative_trace_id、可选 llm_summary) |
| ③ GENERATE (mutate) | `mining/seed_extractor.py::SeedExtractor.extract(cluster, trace_rows)` → `FailureSeed`;`mutation/base_operator.py::MutationOperator(ABC).generate(seed_scene_config, seed_instruction, parent_task_id, benchmark)`;算子:`PosePerturbation`、`DistractorAdd`、`InstructionParaphrase`(LLM gpt-4o-mini)、`MaterialSwap`、`HorizonExtension`(参数见 `configs/mutation.yaml`,max_candidates_per_cycle=50) | seed → `CandidateTask`(schema.py,含 mutation_type/mutation_params/embedding/promotion_evidence)→ `memory.record_candidate_task()` |
| ④ VALIDATE+PROMOTE | `promotion/pipeline.py::PromotionPipeline.run(candidate)` 顺序执行五道闸门(`promotion/gates.py`):`SolvabilityGate(n_trials=10, min_sr=0.5)` → `ReproducibilityGate` → `RedundancyGate(similarity_threshold=0.85, 余弦相似度)` → `DiscriminativePowerGate(n_trials=20, min_spread=0.2)` → `HumanReviewGate`(入队 `human_review_queue.jsonl`,`promotion/human_review_queue.py`);每闸门经 `ThreadPoolExecutor` 提交并施加 `gate_timeout_minutes=120` 超时,**超时/异常→deferred(不判失败)**;结果状态 promoted/rejected/archived/deferred + gate-by-gate evidence 写回 `tasks.promotion_evidence` | candidates → 状态更新 `memory.update_task_promotion_status()` |
| ⑤ MONITOR | `EvolutionLoopOrchestrator._check_saturation()` → `EvalMemory.get_saturated_tasks(threshold=0.95)`(所有模型 SR≥阈值即饱和,准备 archive) | promoted tasks → tasks_saturated 计数 |
| ⑥ REPORT | `reporting/report_generator.py::ReportGenerator.generate(output_path, cycle_result)` + `reporting/heatmap.py` | → `eval_memory/report_<cycle_id>.md` |
| (export) | 闭环外由 CLI 触发:`exporter/hard_case_exporter.py::HardCaseExporter.export()`,`_write_lerobot_format` / `_write_jsonl_format` | 失败 episode → LeRobot/JSONL 数据集,回灌持续学习(closed-loop eval→data→training) |

**Critics**(为 gate/标注提供打分):`critics/semantic_critic.py::SemanticCritic.judge()` — 优先 `_call_local_judge()` POST `http://{host}:{port}/judge`(复用 policy server),若 model_id 命中 `circular_bias_patterns`("Qwen*",防被评模型自评的循环偏置)则 `_call_gpt4o_mini()` 兜底;另有 `safety_critic.py`(torque_spike/grasp_retry/max_episode_length 阈值)与 `robustness_critic.py`(sensitivity_threshold=0.15)。

**遥测**:每步 `_log_step()` 追加 `eval_memory/evolution_loop_log.jsonl`;循环结束 `_write_metrics()` 写 `sepa_eval_metrics.json`(traces_written、candidates_generated、promotion_yield、gate_timeout_count 等)。

### EvalMemory 持久层(`memory/eval_memory.py`)
- SQLite(`eval.db`,`_apply_ddl()` 逐条执行 DDL,含 `PRAGMA journal_mode=WAL; busy_timeout=5000`)+ msgpack 文件存储(`eval_memory/{run_id}/{trace_id}.msgpack`)。
- `record_trace()` crash-safe 双写:写 `.tmp` → `os.replace` 原子重命名 → DB insert;DB 失败则补偿删除文件;`fsck()` 清理孤儿文件;`prune(retention_days=90)` 只删旧 trace 文件保留 DB 行。
- 查询 API:`get_failures / get_failures_by_cluster_window / get_saturated_tasks / get_discriminative_tasks / update_critic_score / record_candidate_task / update_task_promotion_status / register_model`。

---

## 4. CLI 入口:`sepa_eval/__main__.py`

`python -m sepa_eval <command>`,argparse 子命令 → `cmd_*` 函数:

| 命令 | 函数 | 作用 |
|---|---|---|
| `run` | `cmd_run` | 组装 `EvolutionLoopOrchestrator`(FailureClusterer+SeedExtractor+4 个变异算子+5 闸门 PromotionPipeline+ReportGenerator)并 `run_cycle()` |
| `eval` | `cmd_eval` | `memory.register_model()` 注册模型;`--ci_mode` 走 `_ci_check` 回归门禁 |
| `promote` | `cmd_promote` | 单独对 DB 中 `promotion_status='candidate'` 的任务跑 PromotionPipeline |
| `report` | `cmd_report` | 生成 Markdown 报告 |
| `export-hard-cases` | `cmd_export_hard_cases` | `--format lerobot|jsonl` 导出失败 episode |
| `diff` | `cmd_diff` | 两 checkpoint 按任务比较 SR;`--fail-on-regression`(回归>0.05 退出码 1) |
| `sync-models` | `cmd_sync_models` | `registry/models_registry.py::ModelsRegistry.load()/sync_to_db()` 同步 `configs/models.yaml` → DB |
| `prune` | `cmd_prune` | `EvalMemory.prune(retention_days)` |
| `review list/approve` | `cmd_review_list/approve` | 人工审核队列操作 |
| `status` | `cmd_status` | EvalMemory 统计 |

公共 helper:`_resolve_memory_dir()`(`--memory-dir` 或 `SEPA_MEMORY_DIR` 环境变量,默认 `./eval_memory`)、`_make_memory()`、`_load_config()`(YAML)。所有子模块 import 均在 try/except 内,缺依赖时降级为跳过对应步骤。

---

## 5. 配置系统

**两套配置并存**:

A) AlphaBrain 侧(训练/评测):
- 单入口 `AlphaBrain/configs/finetune_config.yaml`(~37KB):顶层 `environment/paths/common/modes`;`modes:` 下 ~39 个模式(如 `qwen_oft`、`libero_eval`、`robocasa_tabletop_eval`、`cosmos_policy`…),每个模式覆写 framework/datasets/training/eval。优先级:model 默认 → dataset 默认 → trainer 默认 → mode 覆写。
- `scripts/parse_config.py::parse_config(config_path, mode)`:解析 YAML、`expand_env_vars()` 支持 `${VAR:-default}`,按 `mode.type=='eval'` 分流,`print` 出 `EVAL_CHECKPOINT/EVAL_BENCHMARK/TASK_SUITE/NUM_TRIALS/EVAL_HOST/EVAL_PORT/EVAL_GPU_ID/EVAL_SERVER_PYTHON/EVAL_CLIENT_PYTHON…` 等 shell 变量,由 `run_eval.sh`/`run_finetune.sh` `eval "$(…)"` 注入。
- `.env`(模板 `.env.example`):`PRETRAINED_MODELS_DIR`、`LEROBOT_LIBERO_DATA_DIR`、`LIBERO_DATA_ROOT`、`LIBERO_HOME`、`LIBERO_PYTHON`、可选 `ROBOCASA_TABLETOP_PYTHON`/`ROBOCASA365_PYTHON`、WANDB;脚本启动前 `set -a; source .env`,YAML 内经 `${oc.env:VAR}`(OmegaConf)引用。

B) SEPA-Eval 侧(`sepa_eval/configs/`):
- `orchestrator.yaml`:full_cycle_schedule=weekly、max_candidates_per_cycle=50、gate_timeout_minutes=120、saturation_threshold=0.95、trace_retention_days=90、n_trials_per_task=5。
- `mutation.yaml`:5 个算子及参数网格(PosePerturbation delta_pos [0.02,0.05,0.10] / delta_rot_deg [5,15,30] 等)、max_variants_per_seed=10、causally_isolate=true;附 `failure_mining.cluster_window.last_n_runs=5`、min_cluster_size=5。
- `critics.yaml`:semantic default=local / fallback=gpt-4o-mini、circular_bias_patterns=["Qwen*"];safety/robustness 阈值。
- `models.yaml`:模型注册表(model_id/framework/checkpoint/benchmarks),经 `sync-models` 灌入 DB。
- 环境变量:`SEPA_MEMORY_DIR`。

---

## 6. 依赖方向图

```
                     ┌──────────────────────────────────────────────┐
                     │ AlphaBrain 评测面 (client-server, WebSocket)  │
                     │  scripts/run_eval.sh ─► parse_config.py      │
                     │   ├─► deployment/model_server/server_policy  │
                     │   │      └─► AlphaBrain/model/framework/*    │
                     │   └─► benchmarks/*/eval/ (仿真客户端)         │
                     └───────────────┬──────────────────────────────┘
              episode 轨迹旁路截获     │            HTTP /judge (语义评审)
                                    ▼            ▲
   sepa_eval.hooks (BenchmarkAdapter/TraceHook)  │
        │ record_trace()                         │
        ▼                                        │
   sepa_eval.memory (EvalMemory + schema) ◄── sepa_eval.critics ──┘
        ▲        ▲          ▲         ▲
        │        │          │         │
   sepa_eval  sepa_eval  sepa_eval  sepa_eval
   .mining    .mutation  .promotion .exporter / .reporting / .registry
        ▲        ▲          ▲         ▲
        └────────┴────┬─────┴─────────┘
                      │ 依赖注入(构造函数传入,不反向 import)
        sepa_eval.orchestrator.EvolutionLoopOrchestrator
                      ▲
        sepa_eval.__main__ (CLI, lazy import + 降级)
```

依赖规则(自内向外):
1. `memory/schema.py` 是最底层公共数据契约,被 hooks/mining/mutation/promotion/exporter 引用;
2. `memory.eval_memory` 只依赖 schema,是唯一持久化门面;
3. mining/mutation/promotion/critics/reporting/exporter 彼此**不直接依赖**,统一通过 orchestrator 的构造注入 + EvalMemory 数据交换解耦;
4. `sepa_eval` → AlphaBrain 只有运行时 HTTP(/judge)与 shell 层(run_eval.sh 产生 trace)边界,无 Python import 依赖;AlphaBrain 完全不知道 sepa_eval 存在(单向)。

---

## 7. 状态与已知缺口(来自 TODOS.md 与代码核对)

- TODOS 中 6 个 pre-release blocker(A2 CandidateTask、A5 /judge、A1 BenchmarkAdapter、A4 crash-safe 双写、T1-T3、O1/O2 遥测)均标记已完成并与代码位置一致。
- 尚未闭合:真实模拟器打通的端到端 evolution loop 验证(CLAUDE.md 明示 "full simulator-backed evolution-loop verification is still pending");`cmd_run` 里 `run_cycle()` 未传 `eval_fn/model_ids`,即 Step 1 EVALUATE 目前假设 trace 已在 memory 中,真实评测须外部经 run_eval.sh + hooks 写入。
- 接口不一致小疵:`EvolutionLoopOrchestrator` Step 4 调 `self._promotion_pipeline.run(all_candidates)`(整列表),而 `PromotionPipeline.run()` 签名是单 candidate 返回 `(status, evidence)`——批量语义靠 orchestrator 的 try/except 容错掩盖,是潜在 bug 点。
- `_write_metrics` 中 `critic_latency_ms` 恒为 0.0(占位)。
- 测试:`sepa_eval/tests/` 14 个测试文件覆盖 memory/mutation/promotion/critics/exporter/reporting/hooks/evolution_loop/integration。
