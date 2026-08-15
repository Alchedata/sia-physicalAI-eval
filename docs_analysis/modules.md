# SEPA-Eval 核心模块深度分析报告

> 目标仓库: `/Users/fei/Code/Alchedata/sea-physicalAI-eval`
> 重点包: `AlphaBrain/sepa_eval/`(约 3,900 行实现 + 2,100 行测试)
> 参考文档: `CLAUDE.md`、`PRD_SEPA_VLA_Eval.md`(v0.2, 1001 行)、`TODOS.md`

SEPA-Eval 是构建在 AlphaBrain VLA 框架之上的**自演化评测系统**,核心闭环为:
**Evaluate → Diagnose(失败挖掘) → Generate(场景变异) → Validate+Promote(五道门) → Monitor(饱和检测) → Report**。
本报告逐模块分析 `hooks/`、`memory/`、`mining/`、`mutation/`、`critics/`、`promotion/`、`orchestrator/`、`exporter/`、`reporting/`、`registry/`,并对照 `TODOS.md` 评估完成度。

---

## 0. 总体数据流

```
 Benchmark 环境 (LIBERO / RoboCasa)
        │ duck-typed env
        ▼
 hooks/LiberoHook · RobocasaHook (BenchmarkAdapter, PROTOCOL_VERSION="1.0")
        │ on_step(obs, action)  →  TraceHook (纯内存累积, 无 I/O)
        ▼ on_episode_end(success)
 memory/EvalMemory  ← record_trace(EpisodeTrace)
        │  SQLite(WAL) traces/tasks/critic_scores/... + msgpack 文件存储
        │  eval_memory/{run_id}/{trace_id}.msgpack (.tmp→rename 崩溃安全)
        ▼ get_failures_by_cluster_window()
 mining/FailureClassifier → FailureClusterer(DBSCAN) → SeedExtractor
        │ FailureSeed (最小复现: scene_config + instruction + causal_factors)
        ▼ operator.generate(seed_scene_config, seed_instruction, ...)
 mutation/ 5 个算子 → list[CandidateTask] → memory.record_candidate_task()
        ▼
 promotion/PromotionPipeline (5 gates, ThreadPoolExecutor + 超时→deferred)
        │ "promoted"/"rejected"/"archived"/"deferred" + evidence dict
        ▼
 orchestrator/EvolutionLoopOrchestrator.run_cycle()  (6 步循环 + JSONL 日志 + metrics)
        ├─→ reporting/ReportGenerator (Markdown 报告 + 热力图)
        └─→ exporter/HardCaseExporter (失败样本 → lerobot parquet / jsonl → 继续学习)

 critics/ (Semantic/Safety/Robustness) 为标注层: 分数经 memory.update_critic_score() 回写
 registry/ModelsRegistry: configs/models.yaml → memory.register_model() (YAML 为真源, DB 为缓存)
```

---

## 1. hooks/ — 基准适配层

**文件**: `hooks/base.py` (226 行)、`hooks/libero_trace_hook.py` (175 行)、`hooks/robocasa_trace_hook.py` (187 行)

### 1.1 `BenchmarkAdapter` 协议 (`hooks/base.py`)
- `@runtime_checkable Protocol`,4 个方法 + 版本常量:
  - `PROTOCOL_VERSION: str`(版本钉死,TODOS ENG-P3 ✅)
  - `reset() -> dict`、`step(action) -> (obs, done, info)`、`get_task_id() -> str`、`get_scene_config() -> dict`
- 设计意图: **新增基准 = 新增一个实现该协议的文件**(CLAUDE.md 关键不变量)。

### 1.2 `TraceHook` (`hooks/base.py`)
- 上下文管理器;episode 循环内只做 `on_step(obs, action)` 内存追加(**严格无 I/O**)。
- `on_episode_end(success)`:
  - `store_obs=False`(默认): 只保留**最后 10 步观测**(failure-window 模式),`failure_step = max(0, step_count - len(window))`;
  - `store_obs=True`: 保留全部观测,并调用 `_compress_images_in_obs_list()` 把 `obs["images"]` 下的 numpy 数组批量 PNG 压缩(`_encode_png`,PIL 缺失时降级 raw bytes)。
  - 组装 `EpisodeTrace(identity, scene, rollout, labels, provenance)` → `memory.record_trace(trace)`。
- `__exit__` 不自动 flush,失败路径语义显式化。

### 1.3 `LiberoHook` / `RobocasaHook`
- 均为 duck-typed 包装(import 时不需要 LIBERO/robosuite),兼容 `(obs, reward, done, info)` 4 元组与 3 元组,丢弃 reward,兜底 `obs -> {"obs": obs}`。
- `LiberoHook.reset()` 处理 LIBERO reset 返回 None 时回退 `env.get_obs()`。
- `RobocasaHook` 额外携带 `env_name`(如 "KitchenTabletop")写入 scene_config。
- 各带一个端到端 helper: `run_libero_episode_with_trace(env, policy_fn, memory, identity, store_obs)` / `run_robocasa_episode_with_trace(...)`,内部创建 adapter+TraceHook,跑完整 episode,从 `info.get("success")` 判成功。
- ⚠️ **已知缺口**: helper 中 `SceneConfig(init_state=b"", replay_mode="reseed")` — 模拟器状态快照未采集(TODOS OQ1 未验证 `env.sim.get_state()` 可确定性回放),这是 Phase 5 e2e 的前置条件之一。

**完成度**: T1/A1/ENG-P3 均 ✅;RoboCasa hook 测试 TST-3 ✅(`tests/test_robocasa_hook.py` 312 行)。`init_state` 采集缺失是明确的 open item。

---

## 2. memory/ — 持久化层

**文件**: `memory/schema.py` (119 行)、`memory/eval_memory.py` (579 行)

### 2.1 schema.py
- `EpisodeTrace` 五段结构: `TraceIdentity`(trace_id/eval_run_id/benchmark/task_id/instruction/model_id/model_version)、`SceneConfig`(scene_config dict + init_state bytes + replay_mode)、`RolloutData`(observations/actions/episode_length/success/failure_step)、`TraceLabels`(critic_scores/failure_type/failure_attribution)、`TaskProvenance`(parent_task_id/mutation_type/promotion_status)。
- `CandidateTask`(TODOS A2 ✅): task_id、parent_task_id、instruction、scene_config、mutation_type/params、promotion_status、embedding(bytes,预留 BGE-M3)、`promotion_evidence: dict`(L2 ✅);工厂方法 `CandidateTask.new()` 生成 uuid。
- `FAILURE_TYPES` frozenset: 8 类规范失败类型(grasp / pose_estimation / language_grounding / contact_dynamics / recovery / out_of_reach / distractor_confusion / timeout)。

### 2.2 `EvalMemory`
- **双写协议**(A4 ✅): `record_trace()` = ①`_pack_trace`(msgpack, numpy→list) 写 `<path>.tmp` → ②`os.replace` 原子改名 → ③DB INSERT → ④DB 失败时补偿删除文件;`fsck()` 递归清理孤儿 `*.tmp`。
- SQLite 配置(ENG-CQ1 ✅): `PRAGMA journal_mode=WAL; busy_timeout=5000`,持久连接 `check_same_thread=False`。
- 6 张表: `traces`、`tasks`(含 `promotion_evidence` JSON 列)、`critic_scores`、`model_task_results`、`failure_clusters`、`models`。
- 文件布局(T2 ✅): `{memory_dir}/{run_id}/{trace_id}.msgpack`;memory_dir 解析顺序: 显式参数 → `SEPA_MEMORY_DIR` 环境变量(T3 ✅)→ `db_path 同目录/traces`。
- 查询/写入 API: `get_failures(benchmark, model_id, failure_type, eval_run_id, limit)`、`get_failures_by_cluster_window(last_n_runs)`(最近 N 个 distinct eval_run_id 的失败 trace)、`get_saturated_tasks(threshold)`、`get_discriminative_tasks(min_spread)`、`update_critic_score()`(校验 trace_id 存在)、`record_candidate_task()`、`update_task_promotion_status(task_id, status, evidence)`、`register_model()`、`prune(retention_days=90)`(D1 ✅,只删文件保留 DB 行)、`load_trace_file()`。

**完成度**: TODOS 中 memory 相关项(T2/T3/A4/L2/D1/ENG-CQ1)全部 ✅;ENG-P1(SQLite→DuckDB/Parquet 迁移设计)明确推迟到 Phase 4。

---

## 3. mining/ — 失败挖掘

**文件**: `mining/failure_classifier.py` (322 行)、`mining/failure_cluster.py` (175 行)、`mining/seed_extractor.py` (82 行)

### 3.1 `FailureClassifier`(ENG-CQ2 ✅)
- 8 个独立检测器类(`_TimeoutDetector`、`_OutOfReachDetector`、`_GraspDetector`、`_ContactDynamicsDetector`、`_RecoveryDetector`、`_PoseEstimationDetector`、`_DistractorConfusionDetector`、`_LanguageGroundingDetector`),注册于模块级 `_DETECTORS` dict。
- 全部为**启发式规则**(纯 stdlib,无 numpy 依赖):
  - timeout: `episode_length >= max_steps(500)`;
  - out_of_reach: 末 10 个动作平均 L2 范数 > 0.8;
  - grasp: gripper_state/qpos 尾值方差 > 0.1;
  - contact_dynamics: failure_step > 0.7×长度(晚期失败);
  - recovery: 末 10 步窗口内近重复动作(L2 距离 < 0.05);
  - pose_estimation: failure_step < 0.3×长度(早期失败);
  - distractor_confusion: obs 中 `num_distractors > 0`;
  - language_grounding: obs `object_class` 未出现在 instruction 文本中。
- `classify(trace) -> (failure_type|None, failure_step)` 按固定优先级 `_PRIORITY_ORDER` 首个命中者返回;`classify_and_update()` 原地写回 `trace.labels.failure_type` 与 `trace.rollout.failure_step`。
- ⚠️ 局限: 启发式阈值都是硬编码常量,启发式之间存在重叠(如早期失败会同时命中 pose_estimation 与 grasp,靠优先级消歧),没有基于 VLM/学习的分类回退。

### 3.2 `FailureClusterer`
- 9 维嵌入 `embed_trace(trace_row)`: `[failure_step/episode_length] + 8 维 failure_type one-hot`;
- `cluster(trace_rows)`: DBSCAN(sklearn 懒加载, eps=0.5, min_samples=5),`len < min_samples` 直接返回 [];噪声点排除;簇内 failure_type 多数投票为 dominant type;代表 trace = failure_step 最接近中位数者;输出 `FailureCluster(cluster_id, failure_type, member_trace_ids, representative_trace_id, llm_summary=None)`。
- `filter_by_window(trace_rows, last_n_runs)` 提供内存内的窗口过滤(与 `EvalMemory.get_failures_by_cluster_window` 的 SQL 版功能重叠)。
- ⚠️ `llm_summary` 恒为 None — CLAUDE.md 提到的 "LLM cluster summarizer"(复用 /judge)在 mining 包内**不存在对应实现文件**,DB 表 `failure_clusters` 也没有写入路径(orchestrator 不落库 cluster)。

### 3.3 `SeedExtractor`
- `extract(cluster, trace_rows) -> FailureSeed`: 定位代表 trace 行,拼出 `FailureSeed(trace_id, benchmark, task_id, task_instruction, failure_type, failure_step, scene_config, causal_factors)`;causal_factors 仅为 `{failure_type: 1.0}` 的均匀权重(PRD 中的多因子归因未实现)。
- ⚠️ `scene_config` 取自 `rep_row.get("scene_config", {})`,但 **`traces` 表没有 scene_config 列**(scene_config 存在 msgpack 文件里),因此从 `get_failures*` 返回的 DB 行走这条路时 seed.scene_config 恒为 `{}` — 除非调用方自行 hydrate。这是 orchestrator 数据流的一处实际断点(变异算子会在空 dict 上操作)。

**完成度**: 分类器/聚类/种子提取骨架完整、测试充分(test_failure_classifier 329 行, test_failure_cluster 107 行);LLM 摘要、真实 scene_config 注入、多因子归因未完成;ENG-P2 (FAISS) 按计划推迟。

---

## 4. mutation/ — 场景变异引擎

**文件**: `mutation/base_operator.py` (76 行) + 5 个算子

### 4.1 `MutationOperator`(ABC)
- 子类需设 `name` 类属性并实现 `generate(seed_scene_config, seed_instruction, **kwargs) -> list[CandidateTask]`(kwargs 至少接受 `parent_task_id`、`benchmark`);失败抛 `MutationError`;`_make_candidate()` 便捷工厂委托 `CandidateTask.new()`。

### 4.2 各算子
| 算子 | name | 变异内容 | 产出数量 |
|---|---|---|---|
| `PosePerturbation` | 同名 | 对 `*_pos/_pose` 键各坐标加 `Uniform(±delta_pos)`;`*_rot/_quat` 仅首分量加 `±delta_rot_rad`(简化旋转,非正确四元数复合) | delta_pos(3)×delta_rot(3)=9 个 |
| `MaterialSwap` | 同名 | 覆写 `material/texture/friction_coeff`(从 5×4×5 值池无放回抽样) | n_variants(默认 3) |
| `DistractorAdd` | 同名 | 写入 `num_distractors` 与轻量 `distractors` 列表(`{"category":"same","count":N,"id":i}`,靠下游 loader hydrate) | 每个 count(1,2,3)一个 |
| `HorizonExtension` | 同名 | `max_episode_steps × factor`(1.5, 2.0),指令加 "First, " 前缀(除非已含时序标记) | 每个 factor 一个 |
| `InstructionParaphrase` | 同名 | LLM(OpenAI gpt-4o-mini)生成改写;`[START]/[END]` 定界符防 prompt 注入;Jaccard 词重叠相似度 ≥0.9 的近重复被过滤 | ≤ n_variants(3) |

- ⚠️ **MaterialSwap 与 DistractorAdd 只改 scene_config dict**,真实模拟器注入路径未验证(TODOS OQ2: robosuite XML 注入未确认)——当前所有算子实际是"配置层变异",变异是否在物理仿真中生效取决于尚未实现的 loader。

**完成度**: PRD 定义的 5 类算子全部有实现且测试覆盖(test_mutation 297 行);与真实仿真联动(Phase 3/5)未完成;PosePerturbation 的旋转扰动为承认的简化实现。

---

## 5. critics/ — 三评审器

**文件**: `critics/semantic_critic.py` (253 行)、`critics/safety_critic.py` (189 行)、`critics/robustness_critic.py` (131 行)

### 5.1 `SemanticCritic`
- `judge(frames, instruction, model_id) -> CriticResult(completion, object_correct, collateral_damage, explanation)`。
- 双后端路由:
  - 默认 `_call_local_judge()` → `POST http://{host}:{port=10092}/judge`(PRD §8.3),5xx/网络错误抛 `CriticError`;
  - **循环偏置检测** `_should_use_fallback(model_id)`: model_id fnmatch 命中 `["Qwen*","qwen*"]` 时强制走 `_call_gpt4o_mini()`(避免被评模型与裁判同骨干互吹,对应 OQ7)。
- `_encode_frame()`: bytes 直传 base64;numpy 帧优先 cv2 再 PIL 编 PNG。
- ⚠️ **关键缺口(ENG-J1, 未完成)**: `/judge` 端点在 `AlphaBrain/deployment/model_server/server_policy.py` 中**尚未实现**——本地路径目前必然失败,SemanticCritic 只能在 mock 或 OpenAI fallback 下工作。

### 5.2 `SafetyCritic`
- 纯启发式,无外部模型。`evaluate(trace_row, rollout_actions) -> SafetyCriticResult(unsafe_contact, force_spike, fragile_success, safety_score)`。
- 三个标志: force_spike(`joint_torques/forces` max > 5.0)、fragile_success(成功但用了 >90% 步数预算)、unsafe_contact(episode 末 20% 动作展平方差 > 0.05);
- `safety_score = 1 - (0.3·force_spike + 0.4·unsafe_contact + 0.3·fragile_success)`,clamp [0,1]。

### 5.3 `RobustnessCritic`
- `evaluate(trace_rows) -> RobustnessCriticResult(robustness_score, sensitive_to, variant_breakdown)`。
- 按 `mutation_type` 分组算 SR;baseline = "seed"/"none" 组(缺失时用全组均值);`sensitive_to` = SR < baseline − 0.15 的变异类型;robustness_score = 所有组 SR 均值。

- ⚠️ 集成缺口: 三个 critic **没有被 orchestrator 或 promotion pipeline 调用**——`memory.update_critic_score()` 存在但无自动调用方;critic 目前是可独立调用的库组件,PRD 中"critic 分数进入 promotion evidence"的接线未完成。

**完成度**: 三个 critic 类本身完整、有测试(test_critics 107 行);/judge 服务端(ENG-J1)与 pipeline 集成属 Phase 4/5 待办。

---

## 6. promotion/ — 五道晋升门

**文件**: `promotion/gates.py` (362 行)、`promotion/pipeline.py` (158 行)、`promotion/human_review_queue.py` (135 行)

### 6.1 gates.py
- `GateOutcome` 枚举: PASS / FAIL / DEFER / ARCHIVE / DISCARD;`GateResult(gate_name, outcome, evidence, message)`。
- 五道门(每个都是 `evaluate(candidate, **kwargs) -> GateResult`):
  1. `SolvabilityGate(n_trials=10, min_sr=0.5)` — 需外部 `eval_fn(candidate, model_id, n_trials) -> SR`;任一模型 SR≥0.5 → PASS,否则 **DISCARD**(硬拒)。
  2. `ReproducibilityGate(n_trials=20, min_failure_rate=0.6)` — 至少一个模型失败率 ≥0.6 → PASS,否则 FAIL(噪声失败)。
  3. `RedundancyGate(similarity_threshold=0.85)` — candidate.embedding(packed float32 bytes,`_bytes_to_vec`/`_cosine_similarity` 纯 stdlib 实现)与所有已晋升 embedding 比余弦;≥0.85 → **ARCHIVE**;无 embedding 或无已晋升任务 → PASS(宽松放行)。
  4. `DiscriminativePowerGate(n_trials=20, min_spread=0.2)` — 最好/最差模型 SR 差 ≥0.2 → PASS;不足或模型 <2 → **DEFER**(不拒,等更多数据)。
  5. `HumanReviewGate(queue_path)` — 入队 `HumanReviewQueue` 后**立即 PASS**(异步非阻塞)。

### 6.2 `PromotionPipeline`(A3 ✅)
- `run(candidate, **gate_kwargs) -> (final_status, evidence_dict)`;逐门顺序执行,每门 submit 到 `ThreadPoolExecutor(max_workers=4)` 并 `future.result(timeout=gate_timeout_minutes*60)`;
- 超时或异常 → **"deferred"**(符合"超时延期不判失败"不变量);DISCARD/FAIL → "rejected";ARCHIVE → "archived";全过 → "promoted";evidence 为 `{gate_name: evidence}` 完整审计链。
- ⚠️ `gate_kwargs` 对所有门统一透传:`RedundancyGate.evaluate` 需要 `promoted_embeddings`、其余门需要 `eval_fn`/`model_ids`,但每门只声明自己的参数 + `**kwargs`,统一透传可行,但要求调用方一次性给齐全部键。

### 6.3 `HumanReviewQueue`
- 追加式 JSONL(`./eval_memory/human_review_queue.jsonl`),事件 `queued/approved/rejected` 各成一行(完整审计);`list_pending()` 返回全部条目(名不符实——含已决议条目,由调用方自行重建状态)。CLI `sepa_eval review list/approve` 已接线(D9.3 ✅)。

**完成度**: 门与管线逻辑完整、测试覆盖(test_promotion 117 行);但**没有任何组件生成 embedding**(BGE-M3 编码在 PRD 中提及、代码中不存在),故 RedundancyGate 现实中总是走"无 embedding → PASS"分支;eval_fn 需要真实模拟器(Phase 5)。

---

## 7. orchestrator/ — 演化循环

**文件**: `orchestrator/evolution_loop.py` (352 行)

- `EvolutionLoopOrchestrator(memory, clusterer, seed_extractor, mutation_engine: list, promotion_pipeline, report_generator, config, log_path, metrics_path)`。
- `run_cycle(eval_fn, model_ids, max_candidates) -> EvolutionCycleResult` 六步:
  1. **EVALUATE**: 若给了 eval_fn,对每个 model_id 调 `eval_fn(model_id=..., n_trials=...)`(trace 落库由 eval_fn 内部经 hooks 完成);
  2. **DIAGNOSE**: `memory.get_failures_by_cluster_window(last_n_runs)` → `clusterer.cluster()`;
  3. **GENERATE**: 每簇 `seed_extractor.extract()` → 遍历算子 `operator.generate(...)` → `memory.record_candidate_task()`,受 `max_candidates_per_cycle`(默认 50)限流;
  4. **VALIDATE+PROMOTE**: `self._promotion_pipeline.run(all_candidates)`;
  5. **MONITOR**: `memory.get_saturated_tasks(threshold=0.95)` 计数;
  6. **REPORT**: `report_generator.generate(output_path=report_{cycle_id}.md, cycle_result=result)`。
- 可观测性: `_log_step()` 写 `evolution_loop_log.jsonl`(O2 ✅);`_write_metrics()` 写 `sepa_eval_metrics.json`(O1 ✅: traces_written、candidates_generated、promotion_yield、critic_latency_ms、gate_timeout_count)——但 `critic_latency_ms` 恒为 0.0(占位),gate_timeout_count 从日志行统计的是 validate_promote/error 而非真实门超时。
- 每步 try/except 兜底,单步失败降级不炸整个 cycle。

### ⚠️ 关键接口不一致(重要发现)
- Step 4 调用 `self._promotion_pipeline.run(all_candidates)`(**传候选列表**,期望返回 promoted_ids 列表),而真实 `PromotionPipeline.run(candidate, **gate_kwargs)` 是**单候选**签名且返回 `(status, evidence)` 二元组。`tests/test_evolution_loop.py` 用的是 `_FakePromotionPipeline.run(candidates) -> [ids]` 假对象,掩盖了这一不匹配。**用真实 PromotionPipeline 组装 orchestrator 会把候选列表当单个 candidate 处理,行为错误**——需要一个批量包装层或修改 orchestrator。同时,promoted 结果没有回写 `memory.update_task_promotion_status()`。
- 另: Step 3 的 seed.scene_config 来自 DB 行(见 §3.3),实际为 `{}`。

**完成度**: 骨架 + 遥测完整,TST-1 ✅(299 行测试,但全 mock);与真实 promotion/critics/simulator 的接线是最大缺口,对应 P5-E2E(显式 skip 的 `test_evolution_loop_e2e_libero_mini`)。

---

## 8. exporter/ — 难例导出

**文件**: `exporter/hard_case_exporter.py` (260 行)

- `HardCaseExporter(memory, output_format="lerobot"|"jsonl")`;`export(output_dir, model_id, benchmark, max_episodes=1000) -> {episodes_exported, output_dir, format}`,0 条不抛错。
- 流程: `memory.get_failures()` → 逐行 `memory.load_trace_file(trace_path)` → 组 episode dict(steps=zip(obs, actions),`_coerce_json` 把 ndarray→list、bytes→hex)。
- lerobot 格式(OQ3 ✅): `data/episode_{i:06d}.parquet`(pyarrow;列 episode_id/step/obs_json/action_json)+ `meta/info.json`;pyarrow 缺失时优雅降级为同名 JSONL。jsonl 格式: 单文件 `hard_cases.jsonl`。
- 定位: 闭环飞轮的"eval → data"出口,供 AlphaBrain 继续学习管道消费。
- ⚠️ obs/action 以 JSON 字符串列存 parquet,并非 LeRobot 官方 schema(图像列、fps、episodes meta 等),下游 lerobot 加载器不能直接读——是"lerobot 风格布局"而非兼容格式。

**完成度**: OQ3 ✅、测试 ✅(test_exporter 96 行);格式兼容性为后续风险。

---

## 9. reporting/ — 报告与热力图

**文件**: `reporting/report_generator.py` (281 行)、`reporting/heatmap.py` (138 行)、`reporting/templates/eval_report.md.jinja`

- `ReportGenerator(memory).generate(output_path, cycle_result) -> str`,五节 Markdown 报告: Capability Frontier(model_task_results 表)、Saturation Map、Failure Taxonomy(traces 按 model×failure_type 分组计数)、Evolved Task Summary(seed/candidate/promoted 计数,可被 cycle_result 覆盖)、Cross-Model Failure Heatmap。Jinja2 渲染,缺失时字符串回退。
- `build_task_model_heatmap(model_task_results) -> {models, tasks, matrix, universal_failures, model_specific_weaknesses}`:
  - universal_failures = 所有模型 SR<0.4 的任务;
  - model_specific_weaknesses = 仅该模型失败(SR<0.4)而其他模型 ≥0.4 的任务;
- `render_heatmap_markdown()`: ✅(>0.8)/⚠️(0.4–0.8)/❌(<0.4)/—(无数据)Markdown 表。
- 直接用 `memory._conn` 裸 SQL(与 orchestrator._write_metrics 一样绕过 EvalMemory 公共 API,是内聚性瑕疵)。
- ⚠️ `model_task_results` 表**没有任何写入方**(全代码库无 INSERT 到该表的路径,demo 脚本除外),真实运行时 Capability Frontier 与热力图为空。

**完成度**: TST-2 ✅(test_reporting 280 行);报告本身可用(demo 已产出 report.html),但依赖尚无生产者的聚合表。

---

## 10. registry/ — 模型注册表

**文件**: `registry/models_registry.py` (76 行)、`configs/models.yaml`

- `ModelsRegistry(yaml_path=configs/models.yaml, memory)`;`load()` 解析 YAML `models:` 列表;`sync_to_db()` 逐条 `memory.register_model(model_id, framework, checkpoint, benchmarks)`。
- 设计原则(D9.2 ✅): **YAML 是真源,DB models 表只是缓存**。CLI `sepa_eval sync-models` 接线。
- `sepa_eval diff ModelA ModelB`(CP1 ✅)与 `--ci-mode`(CP2 ✅)在 `__main__.py` 中基于 model_task_results/traces 查询实现。

**完成度**: ✅ 完整,测试在 test_prune_and_registry (218 行)。

---

## 11. CLI(`__main__.py`, 778 行)

子命令: `run`(整循环)、`eval`(注册模型+评测, 含 `--ci-mode`)、`promote`、`report`、`export-hard-cases`、`review list/approve`、`status`、`diff`、`sync-models`、`prune`。`SEPA_MEMORY_DIR` 环境变量在此消费(T3 ✅);PKG-1 声称已使其可安装(`sepa-eval` 入口)。

---

## 12. 完成度总评(对照 TODOS.md)

| 模块 | 完成度 | 说明 |
|---|---|---|
| memory/ | ★★★★★ | 全部 TODO ✅;双写+WAL+prune+fsck 均实现且测试充分 |
| hooks/ | ★★★★☆ | 协议+两 hook+测试 ✅;init_state 快照缺失(OQ1) |
| mining/ | ★★★★☆ | 分类/聚类/种子 ✅;LLM 簇摘要缺失;DB 行无 scene_config 导致种子空配置 |
| mutation/ | ★★★★☆ | 5 算子 ✅ + 测试;仅配置层变异,真实 sim 注入未验证(OQ2) |
| critics/ | ★★★☆☆ | 三 critic 类完整;/judge 服务端未实现(ENG-J1),且无自动接线到管线 |
| promotion/ | ★★★★☆ | 5 门+异步超时+审计 ✅;embedding 生成缺失使 RedundancyGate 实际空转 |
| orchestrator/ | ★★★☆☆ | 6 步骨架+遥测+测试 ✅;**与真实 PromotionPipeline 接口不匹配**、晋升状态不回写、critic 未接入 |
| exporter/ | ★★★★☆ | lerobot/jsonl 双格式 ✅;非严格 LeRobot 兼容 schema |
| reporting/ | ★★★★☆ | 报告+热力图+测试 ✅;依赖的 model_task_results 无生产写入方 |
| registry/ | ★★★★★ | YAML 真源+DB 缓存 ✅ |

**TODOS 状态核对**: 6 个 pre-release blocker(PKG-1/2、TST-1/2/3、LINT-1)均标 ✅;未完成项集中在: **ENG-J1(/judge 服务端)**、**P5-E2E(真实模拟器闭环)**、OQ1/OQ2/OQ7(外部验证)、ENG-P1/P2(规模化推迟项)、DOC-1/2/3、PKG-3(版本号不一致 0.1.0 vs 1.0.1)。

## 13. 核心结论

1. **单模块质量高、闭环接线未完成**: 每个模块独立看实现完整、防御性强(懒 import、优雅降级、崩溃安全),单元测试 ~2,100 行;但模块间真实组装存在 3 处断点——orchestrator↔PromotionPipeline 的 `run()` 签名不匹配(批量 vs 单候选)、seed.scene_config 从 DB 行取不到、critics 无人调用。测试大量使用 Fake 对象,恰好掩盖了这些集成缺口。
2. **系统当前是"demo-complete, e2e-pending"**: 合成 trace demo 可跑通全链路报告;但 Phase 5 真实模拟器闭环(P5-E2E)被显式 skip,前置依赖为 /judge 端点(ENG-J1)、LIBERO init_state 确定性回放(OQ1)、robosuite 材质注入(OQ2)。
3. **RedundancyGate 与聚类摘要为纸面能力**: BGE-M3 embedding 生成与 LLM 簇摘要在 PRD/schema 中有位置,代码中无实现,当前分别退化为"总是 PASS"与恒 None。
4. **可观测性设计到位**: WAL+双写、JSONL 步日志、metrics.json、promotion evidence 审计链、human review 追加式审计,均符合 PRD 不变量;个别指标(critic_latency_ms)为占位。
