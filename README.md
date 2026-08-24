# adaptive-engine

领域无关的**约束序列决策引擎**契约层。

给定「谁 / 有哪些候选 / 历史反馈」，输出「在约束下最优的一组候选 + 理由」。
内核不知道自己在做教育出题、商品推荐还是训练排程——它只解一个数学问题。

`contracts/` 只放**契约**，不放算法；`engine/` 是照契约做的实现。契约先钉死，是防止业务概念渗进内核的唯一手段。

---

## 跑起来

```bash
pip install -r requirements.txt

# key:tenant 从环境变量装载，不进仓库；只存 SHA-256，不存明文
export ADAPTIVE_API_KEYS="dev-key:tenantA"
export ADAPTIVE_DB="adaptive.db"
uvicorn engine.api:create_app --factory --port 8080
```

或直接用镜像（单节点，SQLite 落在挂载卷里）：

```bash
docker build -t adaptive-engine .
docker run -p 8080:8080 -v adaptive-data:/data \
  -e ADAPTIVE_API_KEYS="dev-key:tenantA" adaptive-engine
# 容器 HEALTHCHECK 探 /readyz（真的 ping 库），编排层 liveness 探 /healthz
```

启动参数可以走配置文件（`adaptive.example.yaml` 是带注释的模板）：

```bash
cp adaptive.example.yaml adaptive.yaml     # 或 export ADAPTIVE_CONFIG=/etc/adaptive.yaml
python -m engine.config                    # 打印本进程实际会用的每一项
```

**优先级是"文件 > 环境变量 > 默认值"**，构造函数参数高于两者。文件是声明出来的意图，环境变量只是某个 shell 当时恰好持有的值；反过来的话，运维 profile 里一个漏掉的 `ADAPTIVE_DB` 就能静默压过仓库里 review 过的配置，那这份文件就没有可信度了。密钥不在文件里——`ADAPTIVE_API_KEYS` 与两个 token 仍只从环境变量读，因为配置文件是最容易被提交进仓库的东西。

配的是三类东西：库位置（`db`）、容量（`workers`、`max_concurrency`、`rate_per_sec`/`burst`）、候选生成（`recall_limit`、`recall_tags`、`explore_pool_factor`）。最后一类没有环境变量等价物：它们是"质量换延迟"的取舍，该放在能 review、能 diff、能随镜像一起发的地方，而不是一行 shell 里。



```bash
# 1) 登记候选（tag 可以完全不给，退化为单维模型照样工作）
curl -X POST localhost:8080/v1/items -H "X-API-Key: dev-key" -H 'Content-Type: application/json' \
  -d '{"items":[{"id":"i1","tags":{"algebra":1.0},"attrs":{"kind":"choice","floor":0.25}}]}'

# 2) L0 零配置取结果（响应里每个 user 带一个 decision_id）
curl -X POST localhost:8080/v1/next -H "X-API-Key: dev-key" -H 'Content-Type: application/json' \
  -d '{"user":"u1","count":10}'

# 3) 回传反馈：带上 decision_id 即可，propensity 由服务端自己查出来补上
curl -X POST localhost:8080/v1/signals -H "X-API-Key: dev-key" -H 'Content-Type: application/json' \
  -d '{"signals":[{"user":"u1","item":"i1","outcome":1,
                  "decision_id":"<上一步返回的>","signal_id":"<decision_id>:i1"}]}'
```

或者用自带 SDK（`engine/client.py`，纯标准库单文件，可直接拷进客户工程）：

```python
from engine.client import AdaptiveClient

api = AdaptiveClient("http://localhost:8080", "dev-key")
api.register_items([{"id": "i1", "tags": {"algebra": 1.0}}])

slate = api.next("u1", count=10, goal="practice_weak")
for item in slate.items:
    print(item["id"], item["why"])

slate.report("i1", outcome=1.0)     # propensity 与幂等键都不用你操心
```


验证与评估：

```bash
python -m pytest              # 不变式 / HTTP 契约
python -m engine.eval_mtor    # Believe 层：ECE / AUC / CAT / 降级，全部对齐 oracle
python -m engine.eval_choose  # 端到端：约束 / 结构反转 / propensity / 意图分离
python -m engine.ope          # 离线策略评估器对真值自校验
python -m engine.calibrate    # rho.target 校准（当前结论：合成数据无法校准）
python -m engine.loadtest     # 容量：QPS / p99，读写混合，客户端与引擎两条时钟
```


---

## 铁律


> **内核代码里出现任何业务词汇（学生 / 题目 / 知识点 / 作业 / 商品 / 订单），即为设计失败。**

内核词汇只有四个：`user` · `item` · `tag` · `signal`，状态叫 `profile`。
客户的业务模型通过 `examples/*.adapter.yaml` 映射进来，不进内核。

---

## 两套契约，必须成对看

| 面向 | 文件 | 设计取向 |
|---|---|---|
| 对外 | `contracts/openapi.yaml` | **易用优先**。意图化、形容词旋钮、零配置可跑 |
| 对外 | `contracts/policy.schema.json` | L3 逃生舱，表达力零损失 |
| 对内 | `contracts/core.py` | **表达力优先**。三原语 + 效用泛函，完全抽象 |
| 翻译层 | `contracts/goals.yaml` | 把对外意图翻译成内核参数 |

`goals.yaml` 是整个设计的枢纽。它证明「抽象的内核」和「好用的接口」之间严丝合缝：

```
客户说 goal: "practice_weak"
  → goals.yaml 翻译成 ρ=peak(0.70), γ=0.3, V=mastery_sum(low_mu), Φ=diversify(0.8)
  → 内核照此求解，客户永远不需要知道 0.70 这个数存在
```

**漏抽象的判断标准**：客户为了达成目标必须理解内核数学 → 翻译层失败。

---

## 内核：三个不可再分的原语

```
Believe :  b_t = f(b_{t-1}, o_t)              信念更新（观测 → 后验状态）
Score   :  u(a | b)                            价值估计（状态 + 候选 → 效用）
Choose   :  argmax_{A ∈ F} U(A | b)            约束选集（效用 + 约束 → 动作集）
```

MTOR / IRT / SAKT 都只是 `Believe` 的不同实现，可插拔。

**全部可配置面收敛成一个效用泛函：**

```
U(A | b) = E_b[ Σ_{a∈A} ρ(o_a) + γ·V(b') ] + Φ(A)
```

- `ρ` 对结果的偏好 —— 峰函数 / 单调增 / 单调减 / 阈值
- `γ` **利用状态还是改变状态** —— 0 = 即时回报，>0 = 追求状态提升
- `V` 后继信念价值 —— 掌握度总和 / 负熵 / 信息增益
- `Φ` 集合结构 —— 分散 / 集中 / 目标熵

一个式子覆盖全部场景（详见 `goals.yaml`）：

- **自适应测评 (CAT)** = `ρ=0, γ=1, V=-H(b')` —— 纯信息增益特例
- **补短板** = `ρ=peak(0.7), γ>0, V=Σμ(low_mu), Φ=分散`
- **强长板** = `ρ=单调增, γ=0, Φ=集中`

"补短板 vs 强长板" 不是两套算法，是 `ρ` 与 `Φ` 的取法差异。

---

## 对外：四层渐进披露

```
L0  { user, count }                          零配置，5 分钟接入
L1  + goal: "practice_weak"                  选意图（90% 客户停在这里）
L2  + tune: { difficulty, focus, freshness } 形容词旋钮，无数学符号
L3  + policy_ref                             逃生舱，完整表达力
```

### 三条不可让步的易用性约束

1. **接入无前置建模**。`tag` 可以完全不提供（退化为单维模型照样工作）；`item` / `tag` 首次出现即自动注册。"必须先定义标签体系"会杀掉一半潜在客户。
2. **结果自解释**。响应里的 `why` 是给终端用户看的人话，不是内部分数字典。
3. **失败要软**。数据不足 / user 未知 / 候选为空一律降级返回 + `confidence: "low"` + `fallback_reason`，永不抛错。SaaS 报 500 就是客户的线上故障。

---

## 分层落位

```
L2  Policy    场景 = 一份声明式配置（客户可自建，不改代码）
L1  Adapter   客户业务模型 → 内核本体（声明式映射 / Ingestion API）
L0  Core      四个概念 + 三原语 + 效用泛函
```

SaaS 走 Ingestion API（客户 push）；私有化走 connector 配置（映射客户表）。
L1 从第一天就是接口，SaaS 只是它的一个实现。

---

## 目录

```
contracts/
  core.py              内部三原语 + 本体 + 效用规格（唯一的真理来源）
  goals.yaml           goal/tune → 内核参数的翻译表 + 防误用校验规则
  openapi.yaml         对外 HTTP 契约（有测试守着，不允许与实际路由漂移）
  policy.schema.json   L3 自定义 policy 的 JSON Schema（**文档，非执行中的校验器**：
                       无代码读取，真正拒绝非法 policy 的是 engine/policy.py 装载期校验）
  why.yaml             对外文案：why 模板 + 按 fallback_reason 的 hint，多语言，数据而非代码
engine/
  mtor.py              Believe 的首个实现：多维在线贝叶斯评分，零训练纯 numpy
  transition.py        转移模型：gamma>0 时"作用后状态如何变化"的领域假设（含出处标注）
  scorer.py            Score.value：把 (ρ, γ·V) 接到 MTOR 输出，逐候选可向量化
  chooser.py           Choose.solve：submodular 贪心 + quota/硬约束 + explore + propensity
  predicates.py        约束表达式的安全 DSL（不用 eval，装载期拒绝非法式子）
  policy.py            翻译层：goals.yaml → core.Utility，装载期跑校验规则；L3 文档同规则校验
  store.py             SQLite 持久化 + 多租户强隔离 + 幂等 + 写串行化
  service.py           EngineService：装配三原语，对外只有 decide / observe / profile
  api.py               HTTP 层（FastAPI）：鉴权、限流、渲染，openapi.yaml 的落地
  rendering.py         表现层：机器可读的 reasons → 人话，模板与 locale 来自 why.yaml
  client.py            Python SDK：纯标准库单文件，闭环只需 decision_id
  observability.py     指标 / 结构化日志 / 在线校准监控（滑窗 ECE）
  ope.py               离线策略评估：消费 propensity，含对真值的自校验
  simulator.py         合成学习者（3PL + 学习/遗忘），故意与 MTOR 异构
  eval_mtor.py         Believe 评估台：ECE / AUC / 能力回收 / CAT 收敛 / 无标签降级
  eval_choose.py       端到端评估台：约束满足 / 多样性反转 / propensity / 意图分离
  calibrate.py         rho.target 校准：按预测成功率分桶测真实状态提升
  loadtest.py          容量评估台：读写混合，客户端与引擎两条时钟，--url 可打真实部署
tests/
  test_invariants.py   不变式：约束是过滤器 / propensity 语义 / 次模性 / 向量化 Φ 等价 / 持久化 / 隔离
  test_api.py          HTTP 契约：401/429/400/422、软降级、decision_id 回放、L3、locale、契约不漂移
  test_client.py       SDK：propensity 不经客户之手、重投幂等、服务端告警确实抛到调用方
examples/
  jzjx.adapter.yaml       教育场景映射（K12 出题）
  commerce.adapter.yaml   电商场景映射（偏好放大）
Dockerfile             单节点镜像：非 root、pin 依赖、HEALTHCHECK 探 /readyz
../.github/workflows/ci.yml  CI（在仓库根）：test / evals / docker 三个可归因的 job
```

两个 adapter 样例的作用是**验证抽象是否真的通用**：同一个内核、同一套契约，两个领域只差一份映射和一个 `goal`。接不住其中任何一个，说明抽象没做对。

---

## MTOR 已测结果

`python engine/eval_mtor.py`（Python 3.12 + numpy，纯合成数据，无需真实数据）。

评估原则：**不设人造阈值，一切对齐 oracle**。oracle 是知道生成过程真实概率的模型，它自身远达不到满分——选择题存在猜对下限，那部分是不可约噪声。有意义的只有差距。

- **校准 ECE 0.0165**，oracle 0.0113，Elo 基线 0.0752。超出 oracle 仅 0.005
- **AUC 0.688**，oracle 上限 0.752，Elo 0.677
- **CAT 主动选择：在这个模拟器上测不出优势**。固定 probe 集上，30 题预算内 CAT 的 RMSE 下降 0.0058，随机选题 0.0061——两者在噪声内等价，CAT 没有更早追上。见下文"测量陷阱"第 3 条：把两组的 outcome 随机数配对之后，原先声称的"19 题达到随机 30 题水平"消失了，那是抽样噪声而非算法效果
- **无标签降级**：剥掉全部 tag 退化为单维模型，AUC 仍 0.664（oracle 0.768），未崩溃

### 过程中定位到的一个真实缺陷

首版 ECE 是 0.0549。reliability 表给出了明确signature：低分桶实际成功率系统性高于预测（0.2–0.3 桶偏 +0.118），高分桶基本无偏。这是**缺失基线成功率项**的典型特征。

补上下限项后（`p̂ = c + (1-c)·σ(g(θ-b))`，`c` 从 item 元数据读取），ECE 0.0549 → 0.0165，那个桶的偏差 +0.118 → 近似归零。**这是补模型项，不是调参数**——`c=0` 时公式精确退化回原式。

内核不解释 `c` 为何存在，只从一个不透明属性里读数；把"四选一是 0.25"或"某渠道基础转化 3%"翻译成这个数是 adapter 的职责。

### 仍然存在的差距（诚实记录）

- **AUC 差 0.064**：来自缺失区分度参数（模拟器每题区分度服从对数正态，MTOR 假设一致）。这是下一个可动的杠杆，代价是每题多学一个参数、需要更多曝光。
- **能力回收 Pearson 0.650 / Spearman 0.621**：偏弱。部分来自同一个区分度错配，部分是多 tag 可识别性的固有难度（24 维、每题 1–3 个 tag 带权、每维实际观测约 25 次）。
- **CAT 优势为零，不是"偏小"**：配对随机数之后 CAT 与随机选题在这个模拟器上无法区分。probe RMSE 被区分度错配带来的不可约误差主导，能力估计变好的空间本身有限；要证明 CAT 有用，需要一个区分度被正确建模的模拟器或真实日志。现状是**未证实**，不是已证伪。
- `goals.yaml` 的 `rho.target` / `gamma` / `Φ` 权重**全部仍未校准**，`provenance.status: uncalibrated`。校准完成前不得写入对外文档或 SLA。

### 评估台自身修掉的三个测量陷阱

1. 用"已观测维度"上的隐能力 RMSE 衡量 CAT 收敛是被污染的——比较集随施测题数增长，估计变好时数字反而会上升。改为**施测前固定的 held-out probe 集**上的预测误差。
2. CAT 与随机两组必须面对**同一个候选池与 probe 集**。首版从共享且不断推进的 RNG 取池，两组拿到不同题目，差异里混进了抽样噪声。改为按 user 定死种子，并加断言：两组第 0 步误差必须完全相等。
3. 光配平候选池还不够，**outcome 的随机数也必须配对**。两组即使做对同一道题，首版也各自从推进中的模拟器 RNG 抽一次伯努利，于是"CAT 更快"里混着"CAT 运气更好"。改为按 user 预生成一组 coin，两组用同一枚硬币判定同一道题。这一条修完，CAT 的优势就归零了——**它原本是噪声**。这是这个评估台给出的最有价值的一个否定结论。

---

## 端到端已测结果（Score + Choose）

`python -m engine.eval_choose`。测的是**只有拼起来才存在的性质**，且不设人造阈值：
要么是同一批数据上两个策略的相对比较，要么是与解析值的精确比较。

- **硬约束零违反**：predicate / exclude / quota / max_per_tag 同时生效，k=8 全满足
- **不可行时软降级**：quota 无解 → `confidence: low` + `constraints_unsatisfiable`，不抛错
- **结构项方向可反转**：同一信念、同一候选池、**同一个 goal**（`practice_weak`），只改 `focus` 这一个旋钮：`focus=broad` 命中 12.6 个 tag（主 tag 占比 0.13），`focus=narrow` 命中 9.5 个（占比 0.50）。分散与集中是平级策略，不是权重变号
- **propensity 与实际频率吻合**：4000 次抽样，6 个探索候选的经验入选率全部落在报告值 ±4SE 内
- **意图真的分离**：同一个信念，`practice_weak` 选中 tag 的 μ 均值 −0.079，`more_like_this` 为 +0.322。补短板与强长板同式不同解

### 顺带修掉的一个隐性 bug：propensity 的语义

原实现每个探索槽位现算一次候选池，报 `1/m`（该槽的条件概率）。但 IPS 加权要的是**边际入选概率**，两者相差 `n_explore` 倍。用错会让所有离线评估整体缩放一个常数——它不会报错，只会给出一个看起来像结论的偏差。

改为：探索池**抽样前一次性定死**，无放回均匀抽取，报 `n_explore / m`。这样边际概率可解析计算而非估计。若抽样中途被 quota/tag 上限拒绝，则边际值退化为近似，此时在 `reasons.propensity_exact` 置 0，让做 IPS 的人知道哪些行不能信。

---

## 参数校准：一个否定性结论

`python -m engine.calibrate` 按预测成功率分桶，测每桶带来的真实能力提升：

```
p~0.30  +0.0202        p~0.60  +0.0261        p~0.90  +0.0326   ← argmax，落在网格边缘
p~0.40  +0.0225        p~0.70  +0.0266
p~0.50  +0.0249        p~0.80  +0.0289
```

（每档 ±0.0005，SE 用 `ddof=1`。曲线比首版更干净的原因是修掉了一个状态泄漏：
首版只快照回滚 `ability`，`baseline` 与 `last_practice` 被留在了已练过的状态上，
后面的 band 因此继承了前面 band 的遗忘/巩固效应，人为制造出形状。）

曲线**单调上升到边界**，没有内部极值。原因在模拟器自身：它的学习规则是
`gain ∝ (0.4 + 0.6·outcome)`，成功率越高收益越大，**结构上不可能产生峰**。
给它加一个 desirable-difficulty 项，校准出来的只会是我注入的那个数，纯属循环论证。

所以：**`rho.target` 保持 uncalibrated**。这里被验证的是估计器而不是数值——
它正确地拒绝把一个数据中不存在的峰报成校准结果。真实的 `rho.target` 需要带真状态提升信号的线上日志。

把 0.80 写进 `goals.yaml` 并标成 calibrated，是这个项目最容易犯、也最该避免的一种造假。

---

## 生产化：原先列出的缺口逐项处理结果

`python -m pytest tests/ -q` → 90 passed（invariants 31 / HTTP 契约 47 / SDK 12）。

### 服务与状态

- **HTTP 服务**（`api.py`）：`/v1/next`（含批量 users、`decision_id`、`hint`、`locale`）、
  `GET /v1/decisions/{id}`、`/v1/signals`、`POST|GET /v1/items` 与 `GET /v1/items/{id}`、
  `/v1/policies` 增删查（L3）、`/v1/profile/{user}`（含 export / delete）、`/v1/goals`、
  `/healthz`、`/readyz`、`/metrics`、`/metrics.json`。`EngineService` 对外只暴露
  decide / observe / profile，HTTP 层只做鉴权、翻译、渲染，不含任何决策逻辑——放在这一层
  的代码都是脱离服务器就测不到的代码。
- **状态持久化**（`store.py`）：SQLite + WAL。beliefs / items / item_params / signals / decisions / predictions 六张表。已验证「换一个 EngineService 实例继续跑，用户状态不重置」。
- **inflate 接上了**：在**读取信念时**按请求时钟施加，不依赖后台扫描。信念最多只陈旧到上一次读取，也就不存在需要运维的定时任务。
- **tag 空间可增长**：新 tag 首次出现即注册；旧信念按当前维度补齐先验，不会因为客户后加标签而失效（有测试守着）。

### 隔离与审计

- **多租户强隔离**：`Store` 的**每个**方法第一个参数都是 tenant，每条语句都带 `WHERE tenant = ?`。没有能跨租户读的方法，也就没有会忘记加条件的调用点——靠结构而不是靠纪律。测试覆盖「同一个 user_id 在两个租户下互为陌生人」。
- **决策可复现**：`model_version` + `policy_id`（goal+tune 的内容哈希）落到每个 Decision 和 `decisions` 审计表。改 tune 会得到不同的 policy_id。
- **API Key 只存 SHA-256**，常数时间比较；key 从环境变量装载，不进仓库。

### 摄入正确性

- **幂等 + 串行化用同一个机制**：`BEGIN IMMEDIATE` 事务内完成「signal_id 抢占 → 读信念 → 更新 → 写回 → item 参数落盘」。重复投递是 no-op（测试验证 30 条重放全部计入 duplicates），并发信号不会丢更新，且这个串行化跨进程有效而非仅跨线程。
- **批处理只读一次目录**：500 条信号原先会产生 1000 次查询。

### 可观测性

- **在线校准**：服务出去的每个 p_hat 落 `predictions` 表，反馈回来时取出并消费，喂给滑窗 ECE。这样漂移不需要跑批就能看见——ECE 是当初选模型的依据（0.0165 vs Elo 0.0752），它的回退才是要盯的东西。
- 结构化 JSON 日志 + 计数器 + 延迟 p50/p95/p99。`alert_threshold` **故意留空**：出厂阈值就是又一个凭空的常数，应该由租户自己的基线定。
- **`/metrics` 输出 OpenMetrics 文本**，Prometheus 直接抓，不需要中间转换器；`/metrics.json` 保留 reliability 分桶表——那张表才说明校准是**怎么**漂的，而它塞不进 OpenMetrics 的平坦结构。抓取需要鉴权（配了 `ADAPTIVE_METRICS_TOKEN` 就用 Bearer，否则用 API Key）：按租户打标的计数器本身就是租户信息。
- **指标是进程级的**。N 个 worker 就有 N 份局部视图，在线校准窗口也被切成 N 份。这是限制而不是设计，写在这里而不是让运维自己踩。

### 运维接口

- **探活与就绪分开**：`/healthz` **不碰数据库**——如果碰，一条慢查询会让进程被杀掉，而不是仅仅被摘流量；`/readyz` 真的 ping 库，失败返回 503。容器 HEALTHCHECK 探后者。
- **schema 迁移**：`MIGRATIONS` 编号列表 + `schema_version` 表，迁移执行器带数据回填钩子（v2 的 `item_tags` 倒排索引就是从既有 `items.tag_weights` 回填出来的）。升级不需要手工 SQL。
- **保留策略**：`POST /v1/admin/purge` 按 TTL 清 `predictions` / `signals` / `decisions`。审计表无上限增长是单节点部署最先撑爆磁盘的地方。
- **数据主体权利**：`GET /v1/profile/{user}/export` 导出该用户全部留存数据，`DELETE /v1/profile/{user}` 删除。响应里明确写着 item 级参数（难度、区分度）是**总体统计量**、不回滚——"删除一条记录"和"撤销一份贡献"是两件事，这个区别正是合规评审会问的。
- **API Key 轮换与吊销**：key 落 DB（只存 SHA-256），`issue` / `revoke` 立即生效，不需要重启，环境变量装载的静态 key 仍然可用作引导。
- **读路径并发**：每线程一条 SQLite 连接，读不再互相排队；写仍然统一走 `BEGIN IMMEDIATE`。


### 转移模型（原先只是个空壳）

`transition.py` 把这件事拆成两半，因为两半的可信度完全不同：

- **方差是精确的**。ADF 更新下后验方差有闭式解，不引入任何领域假设。信息增益、CAT、不确定性驱动的探索都建在这上面，安全。
- **均值移动是领域假设**。纯贝叶斯滤波下后验均值是鞅，`E[Δμ]=0`——光观测不可能提升掌握度。任何正的期望增益都来自"作用会改变真实状态"这个领域声明。

所以每个模型显式声明自己关于增益形状 `f(p)` 的假设，并标注状态：

- `info_only` — `exact`，鞅，价值全部来自方差缩减
- `monotone_gain` — `supported`，`f(p)=0.4+0.6p`，calibrate.py 在参考模拟器上确认的形状
- `desirable_difficulty` — `hypothesis`，`4p(1-p)`，**校准未能确认**，选它等于主张一个数据没显示的效应
- `state_resolution` — `exact`，`E|r-p|=2p(1-p)`，关于"估计会移动多少"而非"会变好多少"

顺带修掉一处：`scorer` 里原来的 `4p(1-p)` 是手写的凸包形状，没有推导。现在形状来自声明的转移模型，领域假设停留在配置里可见。

**刻意不提供名为 `default` 的模型**：这种名字会把"这个部署签下了哪个领域假设"藏起来。goals.yaml 里原来的 `transition_model: default` 已全部改成显式名字。

`η`（一次交互值多少）是所有候选共享的正常数，比较时会约掉、并入 `γ`。所以**排序不需要校准 η**，只需要校准 `f(p)` 是哪一个。

### 离线策略评估（`ope.py`）

propensity 记录了两轮，现在有了消费方。因为 chooser 报的是**边际**入选概率，slate 价值可以在 item 级无偏 IPS 估计：

```
V(π_t) = E_l[ Σ_{a∈A_l} (π_t(a)/π_l(a)) · r_a ]
```

自校验结果（均匀策略产日志 → 离线估计目标策略 → 再真跑目标策略比对）：

- `challenge`：估计 4.733 ± 0.507，真值 4.211 ± 0.050，gap +0.522（3 pooled SE = 1.53）
- `more_like_this`：估计 5.911，真值 6.086，gap −0.175（3 pooled SE = 1.80）
- `screen`：估计 3.467，真值 2.931，gap +0.536（3 pooled SE = 1.32）

三者全部落在 3 SE 内。同时报告 `ess`（1200 行有效样本 128–144）、`coverage`（~10%，均匀策略从池里抽 k 个的必然结果）、`clipped` —— 光给点估计的 IPS 基本没用，它会以这三种方式无声失效。

口径声明：三个 goal 是**一族三次检验**，3σ 的族错误率约 0.8%；这里的 "OK" 是 sanity gate，不是无偏性证明——它只说明 gap 相对**本次运行自身**的噪声不大。单次通过是必要条件，不是充分条件。`screen` 的 gap 相对另两个偏大的直接原因是它的 `explore_floor=0.05` 在 k=8 时只换来 1 个探索槽，支撑集最薄。

为此给 chooser 加了 `inclusion_probabilities()`：解析计算每个候选的边际入选概率，不靠采样估计。这里有个易错点：**没被抽到的探索池成员仍然有 `n_explore/m` 的概率**，从单次实现读概率会把它记成 0，从而悄悄缩小目标策略的支撑集。

### 测试

`tests/test_invariants.py` + `tests/test_api.py` + `tests/test_client.py`，每条断言对应一个产品承诺，不是"代码能跑"：

- 把**效用最高**的那个 item 排除掉，它必须消失（若排除是打分惩罚，分数够高就会漏出来）
- propensity 对 3000 次蒙特卡洛的经验入选率；解析边际与采样结果一致
- diversify 的边际收益单调不增（次模性，`(1-1/e)` 保证的前提）
- 负结构权重在装载期被拒；L3 注册时同样被拒，且不落库
- 谓词 DSL 拒掉 `__import__('os').system(...)` 等一切非语法输入
- 缺失属性满足 `!=`（否则可选字段变必填）
- 401 不回落到默认租户；429；坏 goal 是 400 而不是 500；冷启动/空目录/不可行约束一律 200 + fallback_reason
- 只回传 `decision_id` 时 propensity 被补齐（`backfilled_propensity == len(items)`），且导出的 signal 行里确实非空
- 同一 slate 重复上报全部计入 duplicates（幂等键由 SDK 从 `(decision_id, item)` 推出）
- `locale=en` / `zh-CN` / `de`：前两者各自语言，未知 locale 回退而非报错
- `openapi.yaml` 的方法+路径集合与实际路由**双向**无差集

### 过程中发现的一个真实缺陷

`last_seen` 用 `0.0` 当"从未观测"的哨兵，与 Unix 纪元冲突，且冲突是**静默**的——该维度永远不会膨胀方差。已改为 NaN。写测试之前这个 bug 不可见，因为真实时间戳都在 1.7e9 附近。

---

## 依然不做的事（诚实边界）

- **`γ` / `Φ` 权重仍未校准**。机制齐了（OPE 已自校验），但定值需要真实日志；合成数据上搜出来的只会是我自己的假设。
- **候选召回仍是"目录 ≤ recall_limit 时全扫，超出则目标导向召回 + 覆盖切片"**。`item_tags` 倒排索引让它不再是无序截断，但没有向量语义召回；超大目录（百万级）需要真正的 ANN/倒排召回层。响应的 `meta.recall` 里如实标出用了哪种策略、扫了多少。
- **限流是进程内的**。N 个副本下实际额度是配置值的 N 倍。多实例需要共享计数器。
- **指标是进程内的**。多 worker/多副本各报各的局部视图（见容量一节）。
- **SQLite 是单节点答案**。`Store` 的接口窄到换 Postgres 是替换而非重写，但现在没有那个实现。跨副本共享状态也要等这个。
- **`rho.target` 保持 uncalibrated**（原因见上文）。

---

## 易用性：站在调用方视角的 8 处返工

功能齐了不等于好用。下面每一条都是「接口在，但用起来会出错或很烦」的地方，
判断标准统一为：**调用方为了正确使用，是否必须自己维护本该服务端维护的东西**。

1. **闭环只需回传 `decision_id`。** `propensity` 是响应里唯一事后无法重建的数字，
   原先要求客户自己保管 per-item 的 propensity 表——这是最容易丢、丢了纠偏就作废的
   一环。现在 `/v1/next` 每个 user 返回 `decision_id`，上报时带上它，服务端批量查出
   propensity 补齐，并在响应里回报 `backfilled_propensity`。两者都缺时计入
   `missing_propensity` 并给出可读告警：**仍然训练模型，但明确说明这批数据做不了 OPE**。
   `GET /v1/decisions/{id}` 可完整回放当时服务出去的内容。
2. **L3 真的接上了。** `policy_ref` 之前只是 schema 里的一个字段。现在
   `POST /v1/policies` 注册、`GET/DELETE` 管理，注册时用**与内置目录完全相同**的规则
   校验——负的 `structure.weight` 会破坏 (1-1/e) 保证，在有人盯着的注册时刻被拒，
   而不是等到第一个引用它的线上请求。`extends` 可继承内置 goal 再局部覆盖（继承意味着
   基目标日后的修正会被带上，而不是冻结成一份副本）。审计里 `policy_id` 直接就是
   `policy_ref`，比一个指向别处文档的哈希有用。
3. **声明了但没实现的字段，改成明确拒绝。** `believe: bandit`、
   `constraints.exclude_seen_within_days` 都在 schema 里、都没实现。原先被静默忽略——
   客户以为在跑另一个模型/另一套排除规则。现在两者都是 400 且说明替代做法。
4. **文案是数据，不是代码。** `why` 与 `hint` 的模板搬到 `contracts/why.yaml`，
   `engine/rendering.py` 只做取值与回退。租户换语言、换语气不需要改代码重新部署；
   缺键按 key 回退到默认 locale，**部分翻译退化成混合语言而不是在已服务出去的
   决策中途抛 KeyError**。`locale` 支持 `zh-CN` 这类写法（浏览器就是这么发的），
   不认识的 locale 回退而不是报错。
5. **降级不但给原因，还给下一步。** 只返回 `fallback_reason` 会让调用方在
   「重试 / 等一等 / 改请求」之间干瞪眼。现在非 high 时同时返回 `hint`，按
   `fallback_reason` 取。没想清楚的状态**返回 None 而不是编一句通用建议**。
6. **报错写给人看。** 422 不再吐 pydantic 内部结构，与其他 4xx 同一信封
   （`{error, detail, problems}`）。写错的 tune 值给近义提示；写错的 tune **键**
   （`freshnes`）原先被 pydantic 静默丢弃、返回 200 且没有 freshness——现在透传到
   策略层，得到 400 + `did you mean 'freshness'?`。诚实的边界：`medium` → `moderate`
   这种同义词编辑距离只有 0.29，靠字符串距离抓不到，所以**取值列表始终完整打印**，
   近义提示只是真拼错时的额外帮助。
7. **只写不能读的路径补上读回。** 登记是只写的，客户无法确认到底存进去什么、
   tag 怎么解析、有没有 id 撞车覆盖。`POST /v1/items` 现在返回 created/updated 计数；
   `GET /v1/items`（keyset 游标，不用 OFFSET，边写边翻页不跳不重）与
   `GET /v1/items/{id}` 可读回。`GET /v1/goals` 同时给出每个旋钮的**确切取值**、
   可用 locale 与已注册 policy——词汇应该查得到，而不是猜完吃 400。
8. **`ts` 变成可选，单位错误当场提醒。** 上报刚发生的事却强迫每个调用方自己合成
   时间戳，只会招来单位错误。省略即「现在」；给出 > 1e11（像毫秒）会在 `warnings`
   里点出来——**时间单位错会悄悄扭曲基于时间的不确定性增长，不报错，只是结论慢慢变错**。
   缺 `signal_id` 同理回报（无幂等键则重投被重复计数）。

另加一个**单文件纯标准库 SDK**（`engine/client.py`）：拖着 numpy 和 FastAPI 的 SDK
没人愿意 vendor，而「你自己调 HTTP 吧」的结果就是每个接入方重新犯上面第 1 和第 8 条。
`Slate.report()` 自动附 `decision_id`，并用 `(decision_id, item)` 作为幂等键——那本来
就是自然主键，一个 item 在一次决策里只有一个结果，所以重投天然去重。服务端告警通过
`warnings.warn` 抛出来，而不是躺在没人读的响应字段里。

**契约不漂移也变成了测试**：`openapi.yaml` 之前写的是 `PUT /v1/items`（服务端收 POST）、
202 + `new_tags`（服务端返 200 + created/updated）——手工维护的契约一定会漂。现在
`test_openapi_yaml_describes_exactly_the_routes_that_exist` 逐条比对方法与路径，
双向都不允许有差集。

---

## 容量：实测而非估计

`python -m engine.loadtest`。脚本报**两条时钟**——客户端墙上时间（含排队）与服务端引擎时间（取自 `/metrics.json`），两者的差就是排队/争用。下面数字来自 20 核开发机、3000 条目录（超过 `recall_limit`，召回层生效）、count=10、write_share 0.2、16 并发客户端。

当前默认配置（`recall_limit=800`、`max_concurrency=2`）实测：

- **单 worker 105.0 QPS**，客户端 `next` p50 165ms / p99 256ms，引擎侧 decide p50 **7.6ms** / p99 21.1ms。冷路径首个请求要付 import / 目录读 / 页缓存预热，所以 loadtest 有 warmup 段，否则它落进尾部。
- **8 worker 252.3 QPS**，但**尾部明显恶化**（`next` p99 881ms、max 7.1s，引擎 decide p99 672ms）：8 worker × 每个 2 并发已经把 20 核这台机器打满，吞吐还在涨而排队开始失控。要吞吐取 8 worker，要可预测的尾部取更少的 worker——这条不是我推荐哪个，是这台机器上两者确实互斥。

下面是拆解每一步各值多少（都在 16 并发客户端、同一台机器上）：

- **进程内多线程不加吞吐，还会反噬**：decide 是 CPU 绑定的 Python+numpy，一个 GIL；单 worker 线程池放开（默认 40）实测 13.4 QPS、客户端 p99 1858ms，收到 2 并发是 32.5 QPS、p99 688ms——**2.5 倍吞吐、2.7 倍尾部改善**。多出来的并发只是在每次 SQLite 调用处放大 GIL 交接。cap=4 (16.3) 与 cap=8 (13.1) 都不如 cap=2，可用并发的峰值就是 2。
- **容量来自进程**：同一份库文件，1 / 4 / 8 worker 聚合吞吐 13.4 / 79.9 / 142.9 QPS（线程池未收窄时）。相对单 worker 是超线性的，因为单 worker 那一档本身就困在上面那个线程反噬里。
- **两者不叠加**：8 worker 下加不加 `max_concurrency=2` 都是 ~143 QPS（142.9 vs 143.8）——16 个客户端摊到 8 个 worker，每个本来就只有 2 个在途。这个开关是在"每 worker 并发 > 2"时才救命。
- **删掉每请求重算的不变量**（见下一节第 3 条）：1 worker + cap=2 从 32.5 升到 52.5 QPS，8 worker 从 143.8 升到 168.2 QPS。
- **`recall_limit` 2000 → 800**（实测定价见 `ServiceConfig.recall_limit` 的注释）：再从 52.5 升到 **105.0 QPS**，8 worker 从 168.2 升到 **252.3 QPS**，而决策质量对"不截断"的偏移全在 1 SE 内。

配置这些的旋钮（走配置文件或环境变量，见上文"跑起来"；发布入口是 `uvicorn engine.api:create_app --factory`，所以只能从 Python 里构造的限制在部署时等于不存在）：`workers`、`max_concurrency`、`rate_per_sec` / `burst`。三者都是**每进程**的：令牌桶在进程内，8 个 worker 的真实上限是配置值的 8 倍。`recall_limit` 同样在这里配，但它改的是决策质量，别当成纯性能旋钮拧：800 已经是实测的饱和点，再往下（300 档掉 4~8%）要先用穿过 `decide` 的对照定价——`engine.ope` 与 `engine.eval_choose` 都直接喂候选池，看不到这个开关。





三处顺带修掉的性能缺陷（都是"不会报错、只是慢"的那种）：

1. **Φ 的边际收益原来是 Python 逐候选、逐已选项算相似度**，O(k²·n)，占了 decide 约 80% 的时间。改成对整个召回池一次性向量化、并把 max/sum 增量维护（O(k·n)）。选择结果**逐位一致**——保留标量参考实现作为定义与 oracle，`test_vectorised_phi_matches_scalar_reference` 对 diversify / concentrate / balanced / cosine 四种形状逐步断言两者相等。单次 decide 从 ~360ms 降到 ~78ms。
2. **item 参数是逐候选查库**（N+1）：没人答过的题没有参数行，于是池里每个候选每个请求都触发一次查询。改成批量预读后记住"已确认不存在"，冷候选不再反复回库。
3. **每请求重算不变量**。profile 里三个 `ncalls` 都等于 60 次 decide × 2000 个候选，也就是每个候选每个请求重来一遍，而这些量与请求无关：
   - `np.clip(item.difficulty_prior, 0, 1)` 对**单个 float** 调 numpy，占一次 decide 的 18%。换成 `_clip1`，对 NaN 与无穷的行为逐位相同（`min`/`max` 传播 NaN 依赖参数顺序，所以不能裸用）。
   - item 的 tag 分解只依赖 (item, tag space)，两者都跨请求存活，而 `MTOR` 不是——所以 memo 放在 item store 上，由 `catalogue_version` 连带失效，`test_reupserting_an_item_invalidates_its_cached_decomposition` 守着这条。
   - 召回只需要 id：候选的元数据通常已经解码在缓存里，原来却仍在 SQL 里带上两个 JSON blob，等于每次 decide 让 SQLite 读、本进程再解析 ~2000 行去重建自己手上的对象。改成取 id、只回库补缺口；召回结果对 recall_by_tags / sample_items 各参数组合**顺序逐项一致**。

   合计单线程 p50 30.5 → 20.7ms，端到端 1 worker 32.5 → 52.5 QPS，三个 eval harness 数字不变。


---

## 交付物：服务化清单

- **Dockerfile**：`python:3.12-slim`、非 root、pin 死依赖、SQLite 落挂载卷；HEALTHCHECK 探 `/readyz`（真的 ping 库）。默认单 worker——多 worker 会把上面说的进程内限流和指标各乘一份，是要显式选择的取舍，不是默默翻倍。
- **CI**（`.github/workflows/ci.yml`）：三个 job 分开，各自可归因——`test`（import 冒烟 + pytest）、`evals`（四个评估台，README 里每个数字的来源）、`docker`（build → 起容器 → 探就绪 → 鉴权抓一次指标）。
- **loadtest**（`engine/loadtest.py`）：`--url` 可打已运行的实例，这是测多 worker/多副本容量的唯一正确方式。


---

## 状态与下一步

- [x] 对外 / 对内契约，翻译表，两个领域的映射验证
- [x] `Believe` 首个实现 MTOR + 合成模拟器 + 评估台
- [x] `Score.value` / `Choose` / 翻译层 `policy.py`
- [x] 区分度参数（按曝光量门控；已实现但默认关闭 `learn_discrimination=False`，开启后的效果未在评估台上定价，故"AUC 差 0.064 来自缺失区分度"的判断仍然成立）
- [x] 校准机制 + 对合成数据能否校准 `rho.target` 的判定（结论：不能）
- [x] 转移模型插件（含每个假设的出处与状态标注）
- [x] 状态持久化 + inflate 回路 + tag 空间增长
- [x] 多租户强隔离 + 决策版本戳 + 审计表
- [x] 摄入幂等 + 跨进程写串行化
- [x] HTTP 服务 + 鉴权 + 限流
- [x] 可观测性（指标 / 结构化日志 / 在线校准）
- [x] OPE 估计器 + 对真值的自校验
- [x] 不变式 / 契约 / SDK 测试 90 项全通（含 purge 三态、v1→当前版本迁移回填、shutdown 连接释放）
- [x] 运维接口：readiness 探针 / schema 迁移 / 保留策略 / 用户导出删除 / API Key 轮换
- [x] OpenMetrics 指标端点 + 每线程读连接
- [x] 性能：Φ 向量化（360ms→78ms/decide）+ 消除 item 参数 N+1；实测容量（8-worker 33 QPS）
- [x] 交付：Dockerfile + CI（test/evals/docker）+ loadtest 评估台
- [x] 易用性 8 项：decision_id 闭环 / L3 接通 / 文案数据化与多语言 / 可读报错 / 目录读回 / ts 可选 / 降级给 hint / 单文件 SDK
- [x] `openapi.yaml` 与实际路由的一致性纳入测试（此前已漂移）
- [ ] `γ` / `Φ` 权重校准：需要真实日志
- [ ] 语义召回层：目标导向召回 + 覆盖切片换成 ANN/倒排（超大目录前）
- [ ] Postgres Store 实现 + 共享限流 + 跨副本指标聚合（多实例部署前）



