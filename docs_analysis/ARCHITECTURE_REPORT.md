# SEPA-Eval / AlphaBrain 系统架构报告

> 生成方式:4 个并行分析子代理(整体架构 / 核心模块 / agent-tool 调用机制 / Storage)交叉验证汇总。
> 详细分报告见同目录:arch.md、modules.md、agent_tool.md、storage.md

## 1. 系统定位与顶层布局

本仓库实现 **SEPA-Eval**——建立在 **AlphaBrain**(VLA 训练/评测/部署框架)之上的**自进化 VLA 评测系统**。核心思想:评测不是终点检查,而是闭环引擎——观察模型失败 → 生成更难的测试变体 → 验证 → 晋升进入活跃 benchmark 分布。

```
sea-physicalAI-eval/
├── AlphaBrain/                  # VLA 框架(policy server、benchmarks、framework adapters)
│   └── sepa_eval/               # SEPA-Eval 包(~3.9k 行实现 + ~2.1k 行测试)
├── eval_memory/                 # 真实运行数据目录(当前基本为空)
├── demo/demo_eval_memory/       # 200 条合成 trace 的演示数据
├── PRD_SEPA_VLA_Eval.md / TODOS.md / paper/
```

**关键耦合设计**:`sepa_eval` 对 AlphaBrain **零 Python import 依赖**,仅通过两条窄接口接入:
1. HTTP `POST /judge`(policy server 端点,复用 VLM 做语义评审)
2. Duck-typed `BenchmarkAdapter` 协议(4 方法:`reset` / `step` / `get_task_id` / `get_scene_config`)——新增 benchmark = 一个新 hook 文件

依赖方向严格单向:`sepa_eval → (HTTP/protocol) → AlphaBrain`,AlphaBrain 完全不知道 sepa_eval 存在。

## 2. AlphaBrain 评测面:Client-Server WebSocket 架构

```
run_eval.sh ──parse_config.py(YAML mode 解析→shell 变量)──┐
   │                                                      │
   ├─ SERVER_PYTHON → server_policy.py                     │
   │     BaseFramework.from_pretrained(QwenOFT/QwenPI/ACT…)│
   │     WebsocketPolicyServer(msgpack-over-websocket)     │
   │                                                      │
   └─ LIBERO_PYTHON / ROBOCASA_PYTHON → benchmarks/*/eval/ │
         eval 客户端(运行模拟器)                            │
         *2model_interface.py ModelClient                  │
           (action chunk 缓存、q99 反归一化、gripper 二值化)  │
```

- **多 conda 环境隔离是该架构的根本动机**:模型依赖(SERVER_PYTHON)与各模拟器依赖(LIBERO/RoboCasa)不可共存,故用 WebSocket + msgpack 跨进程通信。
- 客户端仅在**建连时**重试(600s 总时长 / 2s 间隔);单次推理请求**无超时、无重试**。
- 配置系统:`configs/finetune_config.yaml` 单入口,`modes:` 块覆盖,`.env` 经 `${oc.env:VAR}` 注入;优先级 模型默认 → 数据集默认 → trainer 默认 → mode 覆盖。

## 3. SEPA-Eval 自进化闭环(核心业务流)

`EvolutionLoopOrchestrator.run_cycle()` 实现 6 步循环,每步写 JSONL 步日志 + metrics JSON:

| 步骤 | 模块 | 机制 |
|---|---|---|
| ① Evaluate | hooks/ | TraceHook 包裹 benchmark 运行,产出 EpisodeTrace 写入 EvalMemory |
| ② Diagnose | mining/ | `FailureClassifier`(规则分类失败类型)→ `FailureClusterer`(DBSCAN 聚类)→ `SeedExtractor`(挑选变异种子) |
| ③ Generate | mutation/ | 5 个算子(PosePerturbation / MaterialSwap / DistractorAdd / InstructionParaphrase / HorizonExtension),共同继承 `MutationOperator` ABC |
| ④ Validate + Promote | critics/ + promotion/ | 3 评审器(semantic=VLM、safety/robustness=纯启发式)+ 5 道晋升闸门 |
| ⑤ Monitor | orchestrator | 饱和检查(`_check_saturation`),SR 过高任务归档 |
| ⑥ Report | reporting/ | Markdown 报告 + 失败热力图 |

CLI 入口 `python -m sepa_eval`,共 12 个子命令(run / eval / promote / report / export-hard-cases / diff / sync-models / prune / review list|approve / status)。

## 4. Agent / Tool 调用机制

**实际存在的 LLM/VLM 调用点仅两处:**
1. `SemanticCritic`:优先 `POST 127.0.0.1:10092/judge`(timeout=60s,失败抛 `CriticError`),含 circular-bias 检测后回退 GPT-4o-mini(JSON-schema vision prompt)。
2. `InstructionParaphrase._call_llm`:指令改写,prompt 用 `[START]/[END]` 包裹做抗注入。

**PromotionPipeline 异步语义**(与 CLAUDE.md 不变量一致):
- 5 个 gate 顺序 submit 至 `ThreadPoolExecutor(4)`,`future.result(timeout=120min)`
- **超时/异常 → deferred(推迟,不淘汰)**;DISCARD/FAIL → rejected;ARCHIVE → archived;HumanReviewGate 入队即 PASS(进入人工审核队列)
- 每次判定留 evidence 审计链

## 5. Storage 存储层

**双层 crash-safe 设计**:
- **SQLite 元数据层**(`eval.db`):`PRAGMA journal_mode=WAL; busy_timeout=5000`;6 张表(traces / tasks / critic_scores / model_task_results / failure_clusters / models)
- **msgpack 文件层**(`traces/`):完整 episode 数据(观测、动作、场景配置)
- **写协议**:msgpack 写 `.tmp` → `os.replace` 原子重命名 → DB commit → 失败补偿删除;`fsck()` 清理 `.tmp` 孤儿
- **registry**:`models.yaml` 为真源,DB 为缓存(`sync-models` 同步)
- **exporter**:LeRobot parquet(无 pyarrow 时降级 JSONL)与 jsonl 两种格式

## 6. 完成度评估:demo-complete / e2e-pending

TODOS 中 6 个 pre-release blocker 均已完成,单模块实现完整、防御性强、崩溃安全设计到位。但**真实模拟器端到端闭环从未跑通**(`eval_memory/eval.db` 除 1 条 model 记录外全空),存在以下经交叉验证确认的问题:

### 🔴 集成断点(3 个子代理独立发现,已在源码中复核)
1. **签名不匹配 bug**:`evolution_loop.py` 和 `cmd_promote` 以 `pipeline.run(候选列表)` 调用,但 `PromotionPipeline.run(self, candidate, **gate_kwargs)` 只接受单个候选;异常被 try/except 吞掉。
2. **SeedExtractor 恒取空配置**:从 DB 行读 `scene_config`,但 traces 表没有该列。
3. **critics 无自动调用方**:三个评审器实现完整但 orchestrator 未接线;测试用 Fake 对象掩盖了这些不匹配。

### 🟡 未落地项
- policy server 的 `/judge` 端点仅客户端侧存在,服务端未实现(ENG-J1)
- LLM 簇摘要恒为 `None`(仅 DB 列存在);RedundancyGate 因无 embedding 生成而空转
- LIBERO init_state 回放(OQ1)、robosuite 材质注入(OQ2)未实现;Phase 5 e2e 测试显式 skip

### 🟡 存储层缺陷
- `traces.trace_path` 存**绝对路径**——demo 数据因目录改名(sia→sea)已全部断链
- 除主键外**无任何二级索引**;critic_scores 无唯一约束(号称 upsert 实为 append)
- `check_same_thread=False` 单连接无锁共享给 ThreadPoolExecutor;无 schema 迁移机制
- fsck 无法回收"rename 后、commit 前崩溃"产生的正式 msgpack 孤儿

## 7. 总体评价

架构设计质量高:窄接口(4 方法协议 + 单 HTTP 端点)、单向依赖、崩溃安全双写、超时 defer 而非 fail 的宽容晋升语义,均体现了对分布式评测系统失败模式的深思。当前状态是**各模块 demo 完成、闭环胶水未验证**——下一步的最高优先级是修复 `PromotionPipeline.run` 调用契约、补 `scene_config` 数据通路、实现服务端 `/judge`,然后跑一次真实 LIBERO 端到端循环。
