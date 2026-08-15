# SEPA-Eval Storage 存储层分析报告

> 分析目标仓库: `/Users/fei/Code/Alchedata/sea-physicalAI-eval`(下文以 `{repo}` 指代;`{base}` = `{repo}/AlphaBrain`)
> 核心代码: `{base}/sepa_eval/memory/`、`registry/`、`exporter/`、`reporting/`
> 参考: `CLAUDE.md`、`PRD_SEPA_VLA_Eval.md`、`TODOS.md`

---

## 1. 总体架构:SQLite 元数据 + msgpack 文件双层存储

`EvalMemory`(`{base}/sepa_eval/memory/eval_memory.py`,约 620 行)是整个 SEPA-Eval 的持久化中枢,采用 **"SQLite 索引层 + msgpack 大对象文件层"** 的双写设计:

```
{memory_dir}/                        # 默认 ./eval_memory,可由 SEPA_MEMORY_DIR 覆盖
├── eval.db                          # SQLite(WAL 模式),6 张表:元数据 + 可查询字段
├── traces/
│   └── {eval_run_id}/
│       └── {trace_id}.msgpack       # 完整 EpisodeTrace(观测/动作/场景快照)
├── human_review_queue.jsonl         # 追加式人工审核队列(HumanReviewQueue)
├── evolution_loop_log.jsonl         # 演化循环逐步日志(EvolutionLoopOrchestrator)
├── sepa_eval_metrics.json           # 循环指标(整文件覆盖写)
└── report_{cycle_id}.md             # 每轮生成的 Markdown 报告
```

数据流(写路径):

```
Benchmark 客户端 (LiberoHook / RobocasaHook, hooks/*_trace_hook.py)
  └─ run_libero_episode_with_trace(env, policy_fn, memory, identity)
       └─ 构造 EpisodeTrace (memory/schema.py 的 dataclass)
            └─ EvalMemory.record_trace(trace)
                 ├─ ① _pack_trace() → msgpack bytes
                 ├─ ② 写 <path>.tmp → ③ os.replace() 原子重命名
                 ├─ ④ _insert_trace_row() INSERT OR REPLACE INTO traces + commit
                 └─ ⑤ DB 失败 → os.remove(final_path) 补偿删除文件
```

数据流(读路径):

```
mining (FailureClusterer)  ← memory.get_failures() / get_failures_by_cluster_window()
mutation / promotion       ← memory.record_candidate_task() / update_task_promotion_status()
exporter (HardCaseExporter)← memory.get_failures() → memory.load_trace_file(trace_path)
reporting (ReportGenerator)← 直接查 memory._conn(traces/tasks/model_task_results)
```

---

## 2. `memory/eval_memory.py` — EvalMemory 详解

### 2.1 连接与 PRAGMA

```python
# EvalMemory.__init__
self._conn = sqlite3.connect(db_path, check_same_thread=False)
self._conn.row_factory = sqlite3.Row
self._apply_ddl()   # 逐条执行 _DDL(按 ';' 拆分),包含:
# PRAGMA journal_mode=WAL;
# PRAGMA busy_timeout=5000;
```

- **WAL 模式**:允许"多读者 + 单写者"并发,读不阻塞写;已在实际数据库验证 `PRAGMA journal_mode` 返回 `wal`(`{repo}/eval_memory/eval.db` 与 `demo/demo_eval_memory/eval.db` 均为 wal)。WAL 是数据库级持久属性,一次设置后续连接沿用。
- **busy_timeout=5000**:写锁冲突时最多自旋 5 秒再抛 `SQLITE_BUSY`,匹配 CLAUDE.md 的强制约束("SQLite must be opened with PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000")。
- `check_same_thread=False`:允许该连接被 promotion 的 `ThreadPoolExecutor` 工作线程复用(见 §8 风险分析)。
- `memory_dir` 解析优先级:显式参数 → 环境变量 `SEPA_MEMORY_DIR` → `dirname(db_path)/traces`(TODOS T3 已完成项)。

### 2.2 msgpack 序列化(`_pack_trace` / `_unpack_trace`)

- `_pack_trace(trace) -> bytes`:将 `EpisodeTrace` 五段结构(identity/scene/rollout/labels/provenance)打包。内部 `_coerce()` 递归地把 numpy `ndarray → list`(用 `type(obj).__name__ == "ndarray"` 鸭子判断,**反序列化端无需 numpy 依赖**),`bytes` 原样走 msgpack 原生二进制,`use_bin_type=True`。
- `_unpack_trace(data) -> EpisodeTrace`:`raw=False` 解码,重建 dataclass;列表不还原为 ndarray。缺省字段有向后兼容默认值(如 `replay_mode="exact"`、`promotion_status="seed"`)。

### 2.3 crash-safe 双写协议(`record_trace`)

注释中明确的 4 步协议(TODOS A4 已完成项):

1. `packed = _pack_trace(trace)`,写入 `<final_path>.tmp`;
2. `os.replace(tmp_path, final_path)` — POSIX 原子重命名,**文件层不可能出现半写文件**;
3. `_insert_trace_row(params)`(`INSERT OR REPLACE INTO traces ... 16 列` + `commit`);
4. DB 插入抛异常时执行补偿写:`os.remove(final_path)`,防止"有文件无 DB 行"的孤儿。

配套的 `fsck()`:用 `glob('**/*.tmp', recursive=True)`(外加 top-level `*.tmp` 兜底)扫描 `memory_dir`,删除崩溃遗留的 `.tmp` 孤儿并返回删除数。这与 CLAUDE.md 不变量 "Trace writes use `.tmp`→rename pattern; `EvalMemory.fsck()` cleans orphans" 一致。

### 2.4 其余公开 API

| 方法 | SQL 目标 | 说明 |
|---|---|---|
| `get_failures(benchmark, model_id, failure_type, eval_run_id, limit=1000)` | `traces WHERE success=0 ...` | 动态拼 WHERE,参数化查询;失败挖掘/导出入口 |
| `get_failures_by_cluster_window(last_n_runs=5)` | `traces` | 先取最近 N 个 distinct `eval_run_id` 再取其中失败 trace(DBSCAN 聚类窗口) |
| `load_trace_file(trace_path)` | 文件 | 读 msgpack → `EpisodeTrace` |
| `get_saturated_tasks(threshold=0.95)` | `tasks` | `saturation_flag=1 OR discriminative_power <= 1-threshold` |
| `get_discriminative_tasks(min_spread=0.2)` | `tasks` | `discriminative_power >= min_spread` |
| `update_critic_score(trace_id, critic_name, score, ...)` | `critic_scores` | 先 `SELECT 1 FROM traces` 校验外键(手工,SQLite 未声明 FK),不存在则 `ValueError`;注:是纯 INSERT,docstring 写 "upsert" 但表无唯一约束,重复打分会累积多行 |
| `record_candidate_task(task: CandidateTask)` | `tasks` | `INSERT OR REPLACE`;`scene_config`/`promotion_evidence` 以 JSON 字符串存储 |
| `update_task_promotion_status(task_id, status, evidence)` | `tasks` | `promotion_evidence = COALESCE(?, promotion_evidence)` 保留旧证据 |
| `register_model(...)` | `models` | `INSERT OR REPLACE`;benchmarks 存 JSON 数组字符串 |
| `prune(retention_days=90)` | 文件 | 删除 `created_at < cutoff` 的 trace **文件**,DB 行保留(元数据可继续查询,故意产生"可接受孤儿行") |
| `close()` | — | 关闭连接 |

时间戳统一由 `_now_iso()` 生成:UTC naive、格式 `%Y-%m-%dT%H:%M:%S.%f`,字符串排序即时间排序(`ORDER BY created_at DESC` 依赖此性质)。

---

## 3. `memory/schema.py` — 数据结构与 SQLite 表结构

### 3.1 Python dataclass(msgpack 侧 trace 格式)

`EpisodeTrace` 由 5 个子结构组成:

- `TraceIdentity`:`trace_id, eval_run_id, benchmark, task_id, task_instruction, model_id, model_version`
- `SceneConfig`:`scene_config: dict, init_state: bytes(模拟器状态快照,用于 replay), replay_mode: "exact"|"reseed"`
- `RolloutData`:`observations: list[dict], actions: list, episode_length: int, success: bool, failure_step: int|None`
- `TraceLabels`:`critic_scores: dict, failure_type: str|None, failure_attribution: dict(如 {"grasp": 0.7})`
- `TaskProvenance`:`parent_task_id, mutation_type, promotion_status ∈ {seed,candidate,promoted,rejected,archived}`

`CandidateTask`(变异引擎产物):`task_id(uuid4), parent_task_id, benchmark, instruction, scene_config, mutation_type, mutation_params, promotion_status, created_at, embedding: bytes|None(BGE-M3 指令向量), promotion_evidence: dict`;工厂方法 `CandidateTask.new(...)`。

`FAILURE_TYPES` 冻结集合(8 类):`grasp, pose_estimation, language_grounding, contact_dynamics, recovery, out_of_reach, distractor_confusion, timeout`。

### 3.2 SQLite 表(DDL 位于 eval_memory.py 的 `_DDL`,共 6 张)

| 表 | 主键 | 关键列 | 用途 |
|---|---|---|---|
| `traces` | `trace_id` | eval_run_id, benchmark, task_id, task_instruction, model_id, model_version, success(INTEGER), failure_step, episode_length, failure_type, promotion_status, parent_task_id, mutation_type, created_at, **trace_path**(指向 msgpack 文件) | trace 元数据索引;失败挖掘/导出的查询入口 |
| `tasks` | `task_id` | benchmark, instruction, scene_config(JSON TEXT), mutation_lineage, promotion_status, discriminative_power(REAL), saturation_flag(INTEGER), promotion_evidence(JSON TEXT), created_at | 候选/晋升任务库 |
| `critic_scores` | 无(堆表) | trace_id, critic_name, score, explanation, confidence, scored_at | critic 打分记录(允许多行历史) |
| `model_task_results` | **(model_id, task_id)** 复合主键 | benchmark, n_trials, success_rate, clean_success_rate, avg_episode_length, last_eval_at | 模型×任务聚合成绩,报告/热力图数据源 |
| `failure_clusters` | `cluster_id` | eval_run_id, failure_type, centroid(BLOB), representative_trace_id, member_count, llm_summary, summarized_at | DBSCAN 聚类 + LLM 摘要结果 |
| `models` | `model_id` | framework, checkpoint, benchmarks(JSON TEXT), created_at | 模型注册缓存(YAML → DB 同步目标) |

**索引现状**:除主键自动索引(`sqlite_autoindex_*`,实测 demo 库中仅有 5 个自动索引)外,**没有任何显式二级索引**。`get_failures` 常用过滤列 `success/benchmark/model_id/failure_type/eval_run_id` 与 `ORDER BY created_at` 均为全表扫描;当前万级以下规模可接受,TODOS 中 ENG-P1(SQLite→DuckDB/Parquet 迁移)与 ENG-P2(FAISS ANN)是已识别的规模化路径。

---

## 4. Trace 数据格式(实际 msgpack 文件验证)

对 `demo/demo_eval_memory/traces/16dcbf19-.../03360ec6-....msgpack` 实际解包(200 条 demo trace,每条约 8.5–10.8 KB):

```
identity : {trace_id, eval_run_id, benchmark: "libero_spatial", task_id: "ls_pick_red_cup_left",
            task_instruction, model_id: "QwenOFT-v2.1", model_version: "2.1.0"}
scene    : {scene_config: {layout:"kitchen_A", seed:1001, objects:[{name,pos}]},
            init_state: bytes[16], replay_mode: "reseed"}
rollout  : {observations: list[15]  # 每步 {agentview_rgb: bytes, wrist_rgb: bytes,
            #                        qpos: list[8], gripper_state: list[1], timestamp}
            actions: list[20]×list[7],  episode_length: 134, success: true, failure_step: null}
labels   : {critic_scores: {robustness:0.846, safety:0.869, semantic:0.827},
            failure_type: null, failure_attribution: {}}
provenance: {parent_task_id: null, mutation_type: null, promotion_status: "seed"}
```

注意:demo 数据为合成(`demo/generate_libero_traces.py`,TODOS DEMO-1),图像 bytes 是占位小块;真实运行中 `store_obs` 由 hook 控制。demo 库 `traces.trace_path` 里记录的绝对路径指向旧仓库名 `sia-physicalAI-eval`(现目录为 `sea-physicalAI-eval`),说明 **trace_path 存绝对路径不可迁移** 是一个实际可见的数据一致性缺陷(demo 库中路径已失效)。

---

## 5. `registry/` — 模型注册表

- `configs/models.yaml`:唯一真源(source of truth),schema:`models: [{model_id, framework, checkpoint, benchmarks: [...]}]`。当前仅注册 `alphabrain_v1`(framework=alphabrain,checkpoint=`checkpoints/alphabrain_v1.pth`,4 个 LIBERO 套件)。
- `registry/models_registry.py::ModelsRegistry`:
  - `load()`:`yaml.safe_load` 解析,文件缺失时 warning + 返回 `[]`(不抛错);
  - `sync_to_db()`:逐条调用 `EvalMemory.register_model()`(`INSERT OR REPLACE INTO models`),缺 `model_id` 的条目跳过;返回同步数。**YAML 为真源、DB models 表仅为缓存**(模块 docstring 明示)。
  - CLI 入口:`python -m sepa_eval sync-models`。
- 实际数据:`{repo}/eval_memory/eval.db` 的 models 表有 1 行(alphabrain_v1),demo 库有 2 行(QwenOFT-v2.1 / NeuroVLA-v1.2)。

---

## 6. `exporter/hard_case_exporter.py` — 硬样例导出格式

`HardCaseExporter(memory, output_format="lerobot"|"jsonl")`,数据流:`memory.get_failures(model_id, benchmark, limit=max_episodes)` → 逐行 `memory.load_trace_file(row["trace_path"])` → episode dict(`episode_id, instruction, model_id, benchmark, task_id, steps:[{obs,action}], failure_type, failure_step, episode_length`)。加载失败仅 warning 跳过,0 条时返回 `episodes_exported=0` 而非抛错。

- **lerobot 格式**(`_write_lerobot_format`):
  ```
  {output_dir}/data/episode_{i:06d}.parquet   # 每 episode 一个文件
  {output_dir}/meta/info.json                 # {format, n_episodes, total_steps, columns, models, benchmarks}
  ```
  parquet 列:`episode_id, step, obs_json, action_json`(obs/action 经 `_coerce_json`:ndarray→list、bytes→hex,再 json.dumps)。**pyarrow 缺失时降级**:写 JSONL 内容但仍用 `.parquet` 扩展名(有意为之,便于下游探测文件;但内容格式与扩展名不符是潜在坑)。
- **jsonl 格式**(`_write_jsonl_format`):单文件 `{output_dir}/hard_cases.jsonl`,每行一个 episode 完整 JSON。
- CLI:`python -m sepa_eval export-hard-cases`。

---

## 7. reporting 读取路径 与 实际目录内容

### 7.1 `reporting/report_generator.py::ReportGenerator`

`generate(output_path, cycle_result=None)` 直接通过 **`self._memory._conn`(触碰 EvalMemory 私有连接)** 执行 4 组查询:

| 报告区块 | 查询 |
|---|---|
| Capability Frontier | `SELECT model_id, benchmark, task_id, success_rate, clean_success_rate, n_trials, last_eval_at FROM model_task_results` |
| Failure Taxonomy | `SELECT model_id, failure_type, COUNT(*) FROM traces WHERE success=0 AND failure_type IS NOT NULL GROUP BY ...` |
| Saturation Map | `SELECT task_id, benchmark, promotion_status, discriminative_power, saturation_flag FROM tasks` |
| Task Summary | `SELECT promotion_status, COUNT(*) FROM tasks GROUP BY promotion_status` |
| Heatmap | `SELECT model_id, task_id, success_rate FROM model_task_results WHERE success_rate IS NOT NULL` → `heatmap.py::build_task_model_heatmap()` |

所有查询均包 try/except,查询失败降级为空数据 + warning(报告永远能生成)。渲染优先 Jinja2 模板 `reporting/templates/eval_report.md.jinja`,缺 Jinja2 时 `_fallback_render` 纯字符串拼接。`heatmap.py` 额外计算 `universal_failures`(所有模型 SR<0.4)与 `model_specific_weaknesses`(仅单模型 SR<0.4),Markdown 渲染用 ✅/⚠️/❌ 分档(>0.8 / 0.4–0.8 / <0.4)。

### 7.2 实际目录内容(实测)

- `{repo}/eval_memory/`:`eval.db`(48 KB,WAL,**所有表为空,仅 models 1 行 alphabrain_v1**)+ 空 `traces/` — 即真实模拟器闭环尚未跑通(与 CLAUDE.md "full simulator-backed evolution-loop verification is still pending" 一致),目前仅执行过 `sync-models`。
- `{repo}/demo/demo_eval_memory/`:`eval.db`(160 KB;traces=200, tasks=26, model_task_results=40, failure_clusters=18, models=2, critic_scores=0)+ `traces/` 下 2 个 run 目录各 100 个 msgpack + `report.md`(ReportGenerator 产物)。
- `{base}/results/`:**不存在**。`scripts/run_eval.sh` 定义 `EVAL_OUT_DIR="results/evaluation/${RESULT_GROUP}/${folder_name}"`,是 AlphaBrain 原生评测(policy server + benchmark client)的输出路径,与 SEPA 的 eval_memory 是两条独立结果通道,尚未有实际产物。

---

## 8. 并发安全与数据一致性评估

**做对的地方:**

1. **WAL + busy_timeout=5000**:读写并发的正确基线;报告生成(读)不会阻塞 trace 写入。
2. **`.tmp` → `os.replace` 原子重命名 + 补偿删除 + `fsck()`**:文件层不会出现半写文件;"文件先行、DB 断言存在性"的顺序保证 DB 行存在 ⇒ 文件存在(崩溃窗口仅在 rename 后、commit 前,产生的是"文件孤儿",可由 fsck 之外的清理策略处理——注意 **fsck 只清 `.tmp`,不清"有文件无 DB 行"的 msgpack 孤儿**,这是一个覆盖缺口)。
3. **全部参数化 SQL**,无注入面;`update_critic_score` 手工校验 trace 存在性弥补了未启用外键约束。
4. **PromotionPipeline 的超时语义**(`promotion/pipeline.py`):每个 gate 提交到 `ThreadPoolExecutor`,超时 → 候选 `deferred` 而非 fail,evidence 记录 `{"timeout": True}`,符合 CLAUDE.md 不变量。
5. **HumanReviewQueue 追加式 JSONL**:决策为新行而非原地改写,天然审计日志;evolution_loop_log.jsonl 同为 append-only。
6. `prune()` 只删文件保留 DB 行,聚合统计不受保留期影响(设计意图明确)。

**风险与缺口:**

1. **单连接多线程共享**:`check_same_thread=False` + 无锁保护的 `self._conn`。若 promotion gate 线程与主循环同时调用 EvalMemory 写方法,Python `sqlite3` 默认隔离下 `execute`/`commit` 交错可能互踩事务(sqlite3 模块有内部 GIL 级串行但 commit 语义可交叉)。当前代码路径基本是单线程写(gate 内主要是 eval_fn 回调),风险可控但未显式防护(无 `threading.Lock`,也未用每线程连接)。
2. **跨进程场景**:多个 `python -m sepa_eval` 进程同时写同一 eval.db 时,WAL 允许但只有 busy_timeout 兜底;`INSERT OR REPLACE` 幂等键(trace_id/task_id 为 uuid)使冲突概率低,尚可接受。
3. **`trace_path` 存绝对路径**:demo 库中已实际失效(指向旧目录名 `sia-physicalAI-eval`),目录迁移/重命名即断链;应改为相对 `memory_dir` 的路径。
4. **无二级索引**:`traces` 上按 success/benchmark/model_id/created_at 的高频查询全表扫描;万级以上应加 `CREATE INDEX idx_traces_success_bm ON traces(success, benchmark, model_id)` 及 `created_at` 索引(TODOS ENG-P1/P2 已列为 Phase 4+ 事项)。
5. **`critic_scores` 无唯一约束**:`update_critic_score` 自称 upsert 实为 append,同一 (trace_id, critic_name) 重复打分产生多行,下游若做 AVG/最新值需自行去重。
6. **`fsck` 覆盖不全**:只处理 `.tmp`;"rename 成功但 commit 前崩溃"留下的正式 msgpack 孤儿、以及"补偿删除失败(OSError 被吞)"的残留文件均不会被回收(prune 按 DB 行驱动同样够不到)。
7. **`sepa_eval_metrics.json` 整文件覆盖写**无 tmp→rename 保护,崩溃可产生半写 JSON(低危,可再生)。
8. **无 schema 版本/迁移机制**:`_DDL` 全部 `CREATE TABLE IF NOT EXISTS`,加列需手工迁移;TODOS ENG-P1 已规划迁移路径设计。

**总体评价**:存储层设计在"单机、单写者、万级 trace 以内"的目标规模下是扎实的——WAL/busy_timeout、原子文件写、补偿删除、fsck、append-only 审计日志等关键一致性机制齐全且与 PRD/CLAUDE.md 不变量对齐;主要技术债集中在孤儿文件回收覆盖面、trace_path 可移植性、二级索引与多线程写防护四点,均属可增量修复,不涉及架构性返工。
