# SEPA-Eval 改进计划(基于 2026-06 架构评估)

> 依据:`docs_analysis/ARCHITECTURE_REPORT.md` 及四份分报告(arch/modules/agent_tool/storage)。
> 原则:先修正确性 bug(P0)→ 打通真实端到端闭环(P1)→ 存储/健壮性加固(P2)→ 功能补全与规模化(P3)。
> 每项标注:验证方式 + 对应 TODOS.md 条目(如有)。

---

## P0 — 正确性修复(阻塞一切真实运行,~1-2 天)

### P0.1 修复 `PromotionPipeline.run` 调用契约不匹配 🔴
- **问题**:`orchestrator/evolution_loop.py` 与 `__main__.py::cmd_promote` 以 `pipeline.run(候选列表)` 调用,但真实签名是 `run(self, candidate, **gate_kwargs) -> (status, evidence)`;异常被 try/except 吞掉,晋升步骤静默失败。
- **方案**(二选一):
  - A(推荐):orchestrator 侧改为逐候选循环调用 `run(candidate, ...)`,聚合 (status, evidence),同时正确注入 `eval_fn` / `model_ids` 等 gate_kwargs;
  - B:在 pipeline 增加 `run_batch(candidates, **kw)` 并保留单候选 `run` 语义。
- **配套**:将吞异常的 `try/except` 改为记录到 `evolution_loop_log.jsonl` 的 ERROR 条目 + cycle result 的 `error` 字段(不再静默)。
- **验证**:新增集成测试——用真实 `PromotionPipeline`(非 Fake)+ 2 个候选跑 `run_cycle()`,断言 `candidates_promoted` 与 DB 状态变化。

### P0.2 修复 SeedExtractor 的 scene_config 数据通路 🔴
- **问题**:`mining/seed_extractor.py` 从 traces 表行读取 `scene_config`,但该列不存在 → 变异种子恒为空配置,整个 Generate 步骤实际无效。
- **方案**:seed 提取时经 `EvalMemory` 加载对应 msgpack trace 文件取 `scene_config`(msgpack 侧已有该字段);或在 traces 表增加 `scene_config` JSON 列(注意与 P2.3 迁移机制配合)。推荐前者(避免双写不一致)。
- **验证**:单测——写入含 scene_config 的 trace,断言 SeedExtractor 输出的 seed 非空且字段完整。

### P0.3 将 critics 接入 orchestrator 🔴
- **问题**:Semantic/Safety/Robustness 三评审器实现完整但无任何自动调用方;Validate 步骤实际只跑 gates。
- **方案**:在 run_cycle 的 Validate 阶段(gate 之前或作为 gate 的输入)对每个候选调用三 critics,将分数写入 `critic_scores` 表并注入 gate evidence;SemanticCritic 不可用时(/judge 未起)降级为 skip + 记录,而非崩溃。
- **验证**:集成测试断言 critic_scores 表有写入、evidence 含 critic 结果。

---

## P1 — 打通真实端到端闭环(核心目标,~1-2 周)

### P1.1 实现 policy server 的 `/judge` 端点(TODOS ENG-J1)
- 在 `deployment/model_server/server_policy.py` 按 PRD §8.3 spec 增加 HTTP `/judge`(与 WebSocket 推理并存,复用已加载的 VLM)。
- SemanticCritic 客户端已实现(POST :10092/judge, timeout=60s),只缺服务端。
- **验证**:启动 server 后 `curl /judge` 冒烟 + SemanticCritic 真实调用测试(可 mock 模型输出)。

### P1.2 LIBERO init_state 回放(PRD OQ1)
- 变异后的 scene_config 需能注入回 LIBERO 环境(init state 设置),否则 mutation 产物无法被真实评测。
- 先支持 PosePerturbation + DistractorAdd 两个算子的回放;MaterialSwap 依赖 robosuite 材质注入(OQ2)可后置。
- **验证**:回放同一 trace 的 init_state,断言物体位姿一致(容差内)。

### P1.3 真实 LIBERO 端到端演练(解除 Phase 5 e2e skip)
- 用小规模配置(1 个 suite、少量 episode)跑完整 `python -m sepa_eval run`:
  eval(经 LiberoHook 写 trace)→ 聚类 → 变异 → critics+gates → 晋升 → 报告。
- 补齐 `model_task_results` 表的生产写入方(当前无人写,报告核心表为空)。
- 落实 `critic_latency_ms` 真实测量(当前占位 0)。
- **验证**:`eval_memory/eval.db` 各表非空、`sepa_eval status`/`report` 输出真实数据;取消 e2e 测试的 skip 标记并入 CI(可标记为 slow/manual)。

### P1.4 推理链路健壮性(AlphaBrain 侧)
- WebSocket 客户端为单次推理请求增加超时 + 有限重试(当前仅建连重试);超时后向 eval 循环返回可恢复错误而非挂死。
- **验证**:kill server 中途的故障注入测试。

---

## P2 — 存储与健壮性加固(~1 周,可与 P1 并行)

### P2.1 trace_path 改为相对路径 🟡
- **问题**:traces 表存绝对路径,目录改名(sia→sea)后 demo 数据全部断链。
- **方案**:DB 只存相对 `memory_dir` 的路径(`{run_id}/{trace_id}.msgpack`),读取时拼接;提供一次性修复脚本迁移 demo 数据;`fsck()` 增加断链检测与报告。
- **验证**:整体移动 eval_memory 目录后 `load_trace` 仍可用。

### P2.2 补齐二级索引与唯一约束
- 索引:`traces(task_id)`, `traces(model_id, success)`, `critic_scores(task_id)`, `model_task_results(model_id, task_id)` 等按查询路径添加。
- `critic_scores` 加 `UNIQUE(task_id, critic_name, model_id)` 使 upsert 真正为 upsert(当前是 append)。
- **验证**:重复写 critic 分数断言只保留最新;EXPLAIN QUERY PLAN 确认索引命中。

### P2.3 轻量 schema 迁移机制(为 ENG-P1 铺路)
- `PRAGMA user_version` + 顺序迁移函数列表(v1→v2→…),EvalMemory 初始化时自动迁移。P2.1/P2.2 的 DDL 变更作为首批迁移落地。
- **验证**:对旧版 demo db 执行迁移的单测。

### P2.4 并发与 fsck 补漏
- ThreadPoolExecutor 场景:为共享 SQLite 连接加线程锁,或每线程独立连接(WAL 支持多读单写)。
- `fsck()` 扩展:清理"rename 成功但 DB commit 前崩溃"的正式 msgpack 孤儿(扫描 traces 目录与 DB 差集)。
- reporting 不再访问 `memory._conn` 私有属性——在 EvalMemory 上暴露只读查询 API。
- **验证**:多线程并发写 trace 压力测试;人为制造孤儿后 fsck 回收。

---

## P3 — 功能补全与规模化(Phase 4+,按需)

- **P3.1 RedundancyGate 落地**:接入 embedding 生成(句向量即可起步),使冗余检测不再空转;为 ENG-P2(FAISS,>10k traces)留接口。
- **P3.2 LLM 簇摘要**:复用 /judge(或 GPT-4o-mini 回退)实现 `llm_summary` 生成,填充 failure_clusters 表既有列。
- **P3.3 robosuite 材质注入(OQ2)**:解锁 MaterialSwap 真实回放。
- **P3.4 SQLite → DuckDB/Parquet 迁移评估(ENG-P1)**:在 P2.3 迁移机制之上做设计文档,>10k traces / 多团队时执行。
- **P3.5 RoboCasa 端到端**:LIBERO 闭环稳定后复制 P1.2/P1.3 流程到 RobocasaHook。

---

## 执行建议

| 阶段 | 内容 | 预估 | 退出标准 |
|---|---|---|---|
| P0 | 3 个正确性 bug | 1-2 天 | 真实组件(非 Fake)集成测试全绿 |
| P1 | /judge + init_state 回放 + e2e | 1-2 周 | `sepa_eval run` 在真实 LIBERO 上完整跑通,DB 各表非空 |
| P2 | 存储加固 | ~1 周(可并行) | 迁移/索引/fsck/并发测试全绿,demo 数据修复 |
| P3 | 补全与规模化 | 按需 | 各项独立验收 |

**测试策略修正**:P0 的根因是测试全用 Fake 对象掩盖了真实接口不匹配——建议新增一层"真实组件组装、仅 mock 外部 I/O(模拟器/LLM)"的契约测试,作为 CI 必跑项,防止同类回归。
