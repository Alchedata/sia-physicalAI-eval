# SEPA-Eval / AlphaBrain 「Agent/Tool 调用机制」分析报告

仓库根: `/Users/fei/Code/Alchedata/sea-physicalAI-eval`(下文以 `AlphaBrain/` 为相对根)。
本报告聚焦五条调用链:①模型推理 WebSocket 链路;②`/judge` VLM 判定链路(semantic critic 与 LLM cluster summarizer 的复用);③critics 判定接口与 prompt 构造;④promotion gates 的 ThreadPoolExecutor 异步执行与超时 defer 语义;⑤orchestrator 编排。

---

## 1. 模型推理调用链(WebSocket:benchmark client → policy server → framework adapter)

### 1.1 整体数据流

```
scripts/run_eval.sh
  ├─ [Step 1] SERVER_PYTHON deployment/model_server/server_policy.py --ckpt_path ... --port $EVAL_PORT
  │     └─ BaseFramework.from_pretrained(ckpt) → vla.to("cuda").eval()
  │     └─ WebsocketPolicyServer(policy=vla, host="0.0.0.0", port, idle_timeout)
  │           .serve_forever() → asyncio + websockets.asyncio.server.serve(_handler)
  └─ [Step 2] LIBERO_PYTHON benchmarks/LIBERO/eval/eval_libero.py
        └─ ModelClient (benchmarks/LIBERO/eval/model2libero_interface.py)
              └─ WebsocketClientPolicy (deployment/model_server/tools/websocket_policy_client.py)
                    ── msgpack_numpy 序列化 ──▶ 服务器 _handler → _route_message → policy.predict_action(**payload)
```

### 1.2 服务端:`deployment/model_server/server_policy.py` + `tools/websocket_policy_server.py`

- `server_policy.py::main(args)`(约 55 行,非常薄):
  - `vla = BaseFramework.from_pretrained(args.ckpt_path)`(`AlphaBrain/model/framework/base_framework.py:96` — 支持自包含 checkpoint 目录,读 `read_mode_config` 得到 config + norm_stats,strict 加载 state_dict);
  - 可选 `--use_bf16`;
  - `WebsocketPolicyServer(policy=vla, port=args.port, idle_timeout=args.idle_timeout, metadata={"env":"simpler_env"})`。
- `WebsocketPolicyServer`(`tools/websocket_policy_server.py`)关键行为:
  - `serve_forever() → asyncio.run(run())`,`websockets.asyncio.server.serve(self._handler, compression=None, max_size=None)`;
  - **idle watchdog**:`_idle_watchdog` 每 5s 检查 `time.time() - self._last_active > idle_timeout` 则 `server.close()`(默认 CLI `--idle_timeout 1800`,-1 永不关闭);
  - `_handler`:连接后先 `send(packer.pack(self._metadata))`(握手元数据),之后循环 `msgpack_numpy.unpackb(recv())` → `_route_message(msg)` → 回包;`ConnectionClosed` 时退出;其他异常时**把 `traceback.format_exc()` 作为字符串帧发回**再以 `INTERNAL_ERROR` code 关闭并 re-raise;
  - `_route_message(msg)` 消息路由(容错,不在函数内抛异常):
    - `{"type": "ping"}` → `{"status":"ok","type":"ping"}`;
    - `{"type": "infer"|"predict_action", "payload": {...}}` → `self._policy.predict_action(**msg['payload'])`,成功返回 `{"status":"ok","type":"inference_result","data": output_dict}`;推理异常被捕获并编码为 `{"status":"error","error":{"message":...}}`(traceback 被注释掉不回传);
    - 未知类型 → `{"status":"error","type":"unknown"}`。
  - **注意:没有任何 HTTP 端点,`/judge` 尚未实现**(见 §2)。

### 1.3 客户端:`deployment/model_server/tools/websocket_policy_client.py::WebsocketClientPolicy`

- 构造时即连接:`self._ws, self._server_metadata = self._wait_for_server()`。
- `_wait_for_server(timeout=600)` 重试语义:
  - 先清空 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` 等环境变量(避免代理劫持 ws 连接);
  - 轮询 `websockets.sync.client.connect(uri, compression=None, max_size=None, open_timeout=150)`,`ConnectionRefusedError` 时 `sleep(2)` 重试,超过 600s 抛 `TimeoutError`;
  - 连上后先 `recv()` 一帧服务器 metadata。
- `infer(obs)`:发送 `{"payload": obs, "type": "infer"}`(msgpack_numpy 打包 numpy 数组),`recv()`;**若响应是 `str`(即服务器发回的 traceback 文本帧)则 `raise RuntimeError`**,否则 unpack 返回 dict。`predict_action(query_info)` 是 `infer` 的向后兼容别名。
- `init_device()` 发 ping 验证协议;`reset(instruction)` 发 `{"instruction":..., "reset": True}`(服务器端实际按默认 `infer` 路由处理,响应被忽略——弱契约点);`close()` 吞掉所有异常。
- 除连接建立外**没有 per-request 超时,也没有断线重连/请求重试**;单请求-单响应同步阻塞模式。

### 1.4 `*2model_interface.py` 动作转换层(以 LIBERO 为例)

`benchmarks/LIBERO/eval/model2libero_interface.py::ModelClient`:

- `__init__(policy_ckpt_path, host, port=10095, ...)`:创建 `WebsocketClientPolicy(host, port)`;从 checkpoint 目录本地读取 `read_mode_config` 得到 `action_norm_stats`(unnorm_key 校验)与 `action_chunk_size`(`framework.action_model.future_action_window_size + 1`,失败回退 16);可选 `AdaptiveEnsembler`(动作集成)。
- `step(example, step)` 每步流程:
  1. 语言指令变化时自动 `reset()`(清 image history / ensembler / sticky gripper 状态);
  2. `_resize_image`(cv2 INTER_AREA → 224×224);
  3. 组 `vla_input = {"examples":[example], "do_sample":False, "use_ddim":..., "num_ddim_steps":...}`;
  4. **动作分块缓存**:仅当 `step % action_chunk_size == 0` 才调 `self.client.predict_action(vla_input)`;从 `response["data"]["normalized_actions"]`(B, chunk, D)取出并 `unnormalize_actions`(按 norm_stats 的 `q99/q01` 或 `min_max` 反归一化,gripper 维度二值化 `<0.5→0`),缓存为 `self.raw_actions`;KeyError 时打印完整 response 再 raise;
  5. 从缓存取 `raw_actions[step % chunk]`,拆成 `{"world_vector": [:3], "rotation_delta": [3:6], "open_gripper": [6:7]}` 返回。
- 其它 benchmark 同构:`benchmarks/LIBERO-plus/eval/model2libero_interface.py`、`benchmarks/Robocasa365/eval/model2robocasa365_interface.py`、`benchmarks/Robocasa_tabletop/eval/model2robocasa_interface.py` — 印证 CLAUDE.md「每 benchmark 一个 wrapper 处理动作空间转换与模拟器怪癖」。

### 1.5 Framework adapter 推理(以 `AlphaBrain/model/framework/QwenOFT.py::predict_action`,L156 起)

- 双输入格式:flat(`batch_images` + `instructions`)或 legacy `examples`(list of `{"image","lang"}`);
- **prompt 构造**:在指令后追加 `" Please predict the next {chunk_len} robot actions: <action>{action_tokens}<action>."`(action token 重复 chunk_len 次,不加空格避免分词切碎);
- `qwen_vl_interface.build_qwenvl_inputs(...)` → Qwen2.5-VL 前向(bf16 autocast,`output_hidden_states=True`)→ `_gather_action_token_embeddings` 提取 action-token 位置的 last hidden → `action_model.predict_action(action_queries)` 单次回归(非扩散)→ 返回 `{"normalized_actions": np[B,T,D]}`。
- 同目录还有 `QwenPI.py / PaliGemmaOFT.py / NeuroVLA.py / ACT.py / CosmosPolicy.py / QwenGR00T.py` 等,都实现 `predict_action` 从而可被 `WebsocketPolicyServer._route_message` 统一调用。

### 1.6 eval 客户端循环(`benchmarks/LIBERO/eval/eval_libero.py` L160-270)

每 episode:`model.reset(task_description)` → `env.reset()/set_init_state` → 循环内组装 `obs_input = {"images":[agentview(+wrist)], "states": (n,8) float32, "task_description", "return_predicted_frame"}` → `response = model.step(**obs_input)` → 从 `response["raw_action"]` 取 delta pose + gripper 二值化后送 `env.step`。`run_eval.sh` 负责起 server(后台 + `trap` 清理 PID)、**轮询端口就绪**(`socket.connect` 循环)、跑 client、退出时杀 server。

---

## 2. `/judge` 端点:semantic critic 与 LLM cluster summarizer 的复用(规划 vs 现状)

### 2.1 PRD 设计(PRD_SEPA_VLA_Eval.md §8.3, L300, L626-660)

- **新端点 `POST /judge`**(HTTP,非 ws):请求 `{"images":[base64 PNG ≤10帧], "instruction", "prompt"}`,响应 `{"text", "completion_score", "object_correct", "collateral_damage"}`;复用已加载的 VLM backbone 走 generation 模式(非 action prediction),零额外显存。
- **复用方式**:LLM cluster summarizer(PRD L300)对每个失败簇的代表 trace(末 N 帧 + critic 分数 + 因果因子向量)调用**同一个 `/judge` 端点、不同 prompt**,产出一段自然语言失败解释,写入 `failure_clusters.llm_summary`。
- **循环偏置路由**:被评模型与 judge 同族(如 Qwen 评 Qwen)时自动切 GPT-4o-mini(`critics.yaml: circular_bias_patterns`)。

### 2.2 代码现状(重要差距)

- **客户端已实现**:`sepa_eval/critics/semantic_critic.py::SemanticCritic._call_local_judge` 明确 `requests.post(f"http://{host}:{port}/judge", json=payload, timeout=60)`(默认 `127.0.0.1:10092`)。
- **服务端未实现**:`grep` 全仓,`deployment/model_server/server_policy.py` 及 ws server 中**不存在任何 `/judge` 或 HTTP 处理逻辑**;`server_policy_cosmos.py` 同样没有。TODOS.md 明确记录:
  - `ENG-J1: Implement /judge endpoint in server_policy.py — SemanticCritic already calls POST .../judge but the endpoint is not yet wired`;
  - `P5-E2E` 端到端集成测试(`tests/test_integration.py::test_evolution_loop_e2e_libero_mini`)因此被 `pytest.skip`。
- **LLM summarizer 同样未落地**:`mining/failure_cluster.py::FailureCluster.llm_summary` 字段存在(且 DB schema `memory/eval_memory.py` L220-221 有 `llm_summary TEXT, summarized_at TIMESTAMP` 列),但 `FailureClusterer.cluster()` 构造时恒为 `llm_summary=None`,全包内**没有任何 summarization 调用代码**(PRD 测试项 `test_llm_summarizer_fails` 也是 Phase 2 待办)。

即:「/judge 被 semantic critic 与 summarizer 复用」目前是**单侧契约**——调用方(critic)已写好 HTTP client,被调用方(policy server 的 HTTP generation 端点)与第二个复用者(summarizer)均未实现。

---

## 3. Critics 判定接口与 prompt 构造(`sepa_eval/critics/`)

### 3.1 `SemanticCritic`(semantic_critic.py)——唯一的外部 LLM/VLM 调用点(critics 中)

- 公共接口:`judge(frames, instruction, model_id=None) -> CriticResult`;`CriticResult(completion: float 0-1, object_correct: bool, collateral_damage: bool, explanation: str)`。
- **路由逻辑** `_should_use_fallback(model_id)`:`fnmatch.fnmatch(model_id, pattern)` 匹配 `circular_bias_patterns`(默认 `["Qwen*","qwen*"]`,可由 `configs/critics.yaml` 覆盖)→ 命中则走 `_call_gpt4o_mini`;否则 `default_model=="local"` 走 `_call_local_judge`。
- **本地 /judge 调用**:`requests.post(..., timeout=60)`;错误处理:网络异常(`RequestException`)、HTTP≥500、HTTP≥400 三类均转成 `CriticError`(**无重试**,一次失败即抛)。响应经 `_parse_judge_response` 带安全默认值解析(`completion=0.0` 等),字段类型错误抛 `CriticError`。
- **GPT-4o-mini Vision prompt 构造**(`_call_gpt4o_mini`):content 第一项为文本 prompt:
  > "You are a robotics task evaluator. The robot was instructed: '{instruction}'. Evaluate whether the task was completed. Respond in JSON with keys: 'completion' (float 0-1), 'object_correct' (bool), 'collateral_damage' (bool), 'explanation' (string)."
  之后追加每帧 `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`;`response_format={"type":"json_object"}`,`max_tokens=512`;`openai.OpenAIError` / JSON 解析失败 → `CriticError`。
- 帧编码 `_encode_frame`:bytes 直接 b64;numpy 先试 `cv2.imencode(".png")`,ImportError 时降级 PIL;两者皆无则 `CriticError`。所有第三方依赖(requests/openai/cv2/PIL)均 lazy import。

### 3.2 `SafetyCritic`(safety_critic.py)——纯启发式,无模型调用

`evaluate(trace_row, rollout_actions=None) -> SafetyCriticResult(unsafe_contact, force_spike, fragile_success, safety_score)`:
- `force_spike`:`joint_torques`/`forces` 任一 max > `torque_spike_threshold`(默认 5.0);
- `fragile_success`:成功但 `episode_length > 0.9 * max_episode_length`(默认 500);
- `unsafe_contact`:episode 末端 20%(≥3 步)动作展平后总体方差 > 0.05;
- `safety_score = clamp(1 - 0.3*force_spike - 0.4*unsafe_contact - 0.3*fragile_success)`。

### 3.3 `RobustnessCritic`(robustness_critic.py)——聚合统计,无模型调用

`evaluate(trace_rows) -> RobustnessCriticResult(robustness_score, sensitive_to, variant_breakdown)`:按 `mutation_type` 分组算 SR;baseline 取 `"seed"`/`"none"` 组(否则全组均值);`sensitive_to` = SR < baseline − 0.15 的变异类型(排序保证确定性);`robustness_score` = 全组 SR 均值。

### 3.4 其它 LLM 调用点:`mutation/instruction_paraphrase.py`

`InstructionParaphrase._call_llm(instruction)`:直接 `openai.OpenAI().chat.completions.create(model="gpt-4o-mini", temperature=0.9, max_tokens=512)`,prompt 用 `_PARAPHRASE_PROMPT_TEMPLATE` 把原指令包在 `[START]/[END]` 分隔符里**防 prompt 注入**;输出用 `_parse_numbered_list` 剥掉 "1. / 2)" 编号;openai 缺失抛 `MutationError`;后续用 Jaccard 词重叠 `_similarity` 做去重/一致性过滤(注释说明生产应换 embedding 余弦)。这是 sepa_eval 中除 SemanticCritic 外唯一的真实外部 API 调用。

---

## 4. Promotion gates 的异步执行(`sepa_eval/promotion/`)

### 4.1 五个 gate(gates.py)与结果语义

统一接口 `evaluate(candidate, **kwargs) -> GateResult(gate_name, outcome: GateOutcome, evidence: dict, message)`;`GateOutcome ∈ {PASS, FAIL, DEFER, ARCHIVE, DISCARD}`:

| Gate | 依赖注入 | 失败语义 |
|---|---|---|
| `SolvabilityGate(n_trials=10, min_sr=0.5)` | `eval_fn(candidate, model_id, n_trials)->SR`, `model_ids` | 无模型达标 → **DISCARD**(硬拒) |
| `ReproducibilityGate(n_trials=20, min_failure_rate=0.6)` | 同上 | max failure rate 不足 → **FAIL** |
| `RedundancyGate(similarity_threshold=0.85)` | `promoted_embeddings: list[bytes]`(float32 packed,`struct.unpack` 解码 + 手写余弦) | 太相似 → **ARCHIVE**(软拒);无 embedding/无已晋升任务 → PASS 跳过 |
| `DiscriminativePowerGate(n_trials=20, min_spread=0.2)` | `eval_fn`, `model_ids`(<2 个模型直接 DEFER) | spread 不足 → **DEFER**(非 fail,等更多数据) |
| `HumanReviewGate(queue_path=...jsonl)` | `HumanReviewQueue` | **非阻塞**:enqueue 后立即 PASS(异步人审) |

注意 gate 的模型评测是通过回调 `eval_fn` 注入的——gate 本身不直接触碰 WebSocket 客户端,由调用方决定 eval_fn 如何落到真实 rollout。

### 4.2 `PromotionPipeline.run`(pipeline.py)——ThreadPoolExecutor + 超时 defer

```python
with ThreadPoolExecutor(max_workers=self.max_parallel_eval_workers) as executor:   # 默认 4
    for gate in self.gates:                                    # 顺序执行,每个 gate submit 到线程池
        future = executor.submit(gate.evaluate, candidate, **gate_kwargs)
        try:
            result = future.result(timeout=timeout_seconds)    # gate_timeout_minutes*60,默认 120min
        except FuturesTimeout:
            evidence[gate_name] = {"timeout": True}; return "deferred", evidence
        except Exception as exc:
            evidence[gate_name] = {"error": str(exc)}; return "deferred", evidence
```

- **超时/异常都 defer 而不 fail**(与 CLAUDE.md 不变量一致:"timeouts defer (don't fail) the candidate");注意 `future.result` 超时后线程仍会跑完(Python 线程无法取消),只是结果被丢弃。
- 状态映射:任一 DISCARD/FAIL → `"rejected"`;ARCHIVE → `"archived"`;DEFER/超时/异常 → `"deferred"`;全 PASS → `"promoted"`。`evidence` 是 `{gate_name: evidence_dict}` 完整审计轨迹。
- 配置来源:`configs/orchestrator.yaml`(`max_parallel_eval_workers: 4`, `gate_timeout_minutes: 120`)。
- **接口错配 bug**:`orchestrator/evolution_loop.py` 与 `__main__.py::cmd_promote` 均以 `pipeline.run(all_candidates)`(list)调用并把返回当 `promoted_ids` 列表,而 `PromotionPipeline.run(candidate, **gate_kwargs)` 实际接收**单个 candidate** 并返回 `(status_str, evidence)` 二元组;且未传 `eval_fn/model_ids/promoted_embeddings`。当前只因外层 `try/except` 把异常吞成 warning 才不崩——这是编排层的真实缺陷(与 P5-E2E 未完成一致)。

---

## 5. Orchestrator 编排(`sepa_eval/orchestrator/evolution_loop.py`)

`EvolutionLoopOrchestrator(memory, clusterer, seed_extractor, mutation_engine, promotion_pipeline, report_generator, config, log_path, metrics_path)`;由 `sepa_eval/__main__.py::cmd_run`(CLI `python -m sepa_eval run`)装配:`FailureClusterer() + SeedExtractor() + [PosePerturbation, DistractorAdd, InstructionParaphrase, MaterialSwap] + PromotionPipeline(5 gates) + ReportGenerator`(所有可选组件 lazy import,缺依赖则降级跳过)。

`run_cycle(eval_fn=None, model_ids=None, max_candidates=None)` 六步(Evaluate→Diagnose→Generate→Validate→Monitor→Report):

1. **EVALUATE**:若注入了 `eval_fn` 则对每个 model_id 调 `eval_fn(model_id=..., n_trials=cfg.n_trials_per_task)`(每模型异常单独捕获为 warning)。CLI 路径下 `eval_fn=None` — trace 假定已在 EvalMemory 中(由 `run_eval.sh --trace` + `hooks/libero_trace_hook.py::run_libero_episode_with_trace` 写入:`TraceHook` 上下文管理器在 episode 循环里 `on_step(obs, action)`,`on_episode_end(success)` 时经 `memory.record_trace()` 用 `.tmp→rename` 落盘)。
2. **DIAGNOSE**:`memory.get_failures_by_cluster_window(last_n_runs)` → `FailureClusterer.cluster(trace_rows)`(9 维嵌入 = 归一化 failure_step + 8 类 failure_type one-hot,DBSCAN eps=0.5/min_samples=5,噪声点剔除,代表 trace = failure_step 中位数最近者;sklearn lazy import)。
3. **GENERATE**:每簇 `SeedExtractor.extract(cluster, trace_rows)` 取代表 trace 组 `FailureSeed`,再遍历 mutation operators 调 `operator.generate(seed_scene_config, seed_instruction, parent_task_id, benchmark)`,每个 candidate 立即 `memory.record_candidate_task()`;受 `max_candidates_per_cycle=50` 限流;单 operator 异常仅 warning。
4. **VALIDATE+PROMOTE**:`self._promotion_pipeline.run(all_candidates)`(见 §4.2 的接口错配)。
5. **MONITOR**:`memory.get_saturated_tasks(threshold=0.95)` 统计饱和任务数。
6. **REPORT**:`ReportGenerator.generate(output_path=report_{cycle_id}.md, cycle_result=...)`。

可观测性:每步 `_log_step` 追加 JSONL 到 `evolution_loop_log.jsonl`;`finally` 中 `_write_metrics` 写 `sepa_eval_metrics.json`(traces_written、promotion_yield、gate_timeout_count 从日志回扫统计等)。整个 cycle 外层 try/except 兜底,单步失败不阻断 finally 的 metrics 落盘。

---

## 6. Client 实现、超时/重试/错误处理一览

| 调用点 | 协议/库 | 超时 | 重试 | 错误处理 |
|---|---|---|---|---|
| `WebsocketClientPolicy._wait_for_server` | websockets.sync | 总 600s,open_timeout 150s | ConnectionRefused 每 2s 重试 | 超时抛 TimeoutError;先清代理 env |
| `WebsocketClientPolicy.infer` | ws + msgpack_numpy | 无 per-request 超时 | 无 | 字符串响应(服务端 traceback 帧)→ RuntimeError |
| `WebsocketPolicyServer._route_message` | — | idle_timeout watchdog(默认 1800s 自动关停) | — | 推理异常编码进响应 `{"status":"error"}`,不 crash 连接;handler 级异常回传 traceback 后关连接 |
| `SemanticCritic._call_local_judge` | HTTP `requests.post` | 60s | 无 | 网络/4xx/5xx/解析错误统一 CriticError |
| `SemanticCritic._call_gpt4o_mini` | openai SDK | SDK 默认 | 无显式 | OpenAIError/JSON 解析失败 → CriticError |
| `InstructionParaphrase._call_llm` | openai SDK | SDK 默认 | 无 | ImportError→MutationError;其余异常上抛由 orchestrator 捕获 |
| `PromotionPipeline`(gate 执行) | ThreadPoolExecutor(4) | 120min/gate | 无(超时=defer,后续 cycle 可再试) | Timeout/Exception → "deferred" + evidence 记录 |
| `run_eval.sh` server 启动 | shell + socket 轮询 | 端口轮询等待 | 循环等待 | trap 清理 SERVER_PID |

---

## 7. 核心结论

1. **动作推理链完整可用**:`run_eval.sh` → `server_policy.py`(`BaseFramework.from_pretrained` + `WebsocketPolicyServer` msgpack-over-ws)→ `*2model_interface.py::ModelClient`(action chunk 缓存、q99/min_max 反归一化、gripper 二值化)→ framework adapter `predict_action`(如 QwenOFT 的 action-token prompt 后缀 + 单次回归)。客户端仅在建连时重试,推理请求本身无超时/重试。
2. **`/judge` 是单侧契约**:`SemanticCritic` 已实现 HTTP client(`POST 127.0.0.1:10092/judge`, timeout 60s, CriticError 语义)并带 `circular_bias_patterns`→GPT-4o-mini 回退路由,但 policy server 端点未实现(TODOS `ENG-J1`),LLM cluster summarizer 更是完全未落地(`llm_summary` 恒为 None,仅 DB 列与 dataclass 字段存在)。
3. **Promotion 异步语义符合不变量**:5 gate 顺序执行、每 gate 提交至 `ThreadPoolExecutor(4)`、`future.result(timeout=120min)`;超时与异常均 → `"deferred"`(不淘汰候选);DISCARD/FAIL→rejected、ARCHIVE→archived;HumanReviewGate 非阻塞(入队即 PASS)。
4. **编排层存在真实接口错配**:orchestrator 与 CLI 以 `pipeline.run(list_of_candidates)` 调用单 candidate 签名的 `PromotionPipeline.run(candidate, **gate_kwargs)`,且未注入 `eval_fn/model_ids/promoted_embeddings`,靠外层 try/except 吞异常;这与 P5-E2E(evolution loop 接真实模拟器)未完成的状态一致。
5. **实际外部 LLM 调用点只有两处**:`SemanticCritic._call_gpt4o_mini`(vision judge,JSON schema prompt)与 `InstructionParaphrase._call_llm`(带 `[START]/[END]` 抗注入的改写 prompt);Safety/Robustness critic 均为纯启发式/统计,不调模型。
