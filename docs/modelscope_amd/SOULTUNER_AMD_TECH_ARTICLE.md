# 从 Qwen3.7 Plus API 到 SoulTuner 35B：在 AMD MI308X 上微调音乐推荐 Planner 的成功实践

> 建议发布标签：`AMD GPU激励计划`、`ROCm`、`大模型微调`、`智能体`、`推荐系统`
> 文章状态：可发布长文初稿。正式发布前补 3 张运行截图和创空间链接即可。
> 项目：SoulTuner — Hybrid RAG × Knowledge Graph × Audio Retrieval × Long-term Memory

## 摘要

音乐推荐并不只是“根据一句话搜几首歌”。同一句“我今天心情很差，想听一些温暖的歌”，同时包含可被知识图谱表达的情绪标签，也包含难以被离散标签完整描述的主观听感。如果 Planner 把它当成纯图谱查询，结果会过度依赖人工标签；如果全部交给向量检索，又会失去歌手、年代、语言、类型和场景等可解释约束。

SoulTuner 当前默认使用 Qwen3.7 Plus API 完成大模型相关任务。为了验证“通用 API 模型”与“领域专用 Planner”在真实项目契约上的差异，我在 AMD MI308X（192 GB HBM）实例上完成了 35B-A3B MoE Planner 的两轮微调，并让两个模型在同一份 canonical sealed 500、同一 V5 system prompt、同一确定性评分器下输出未经修复的原始结果。

微调 35B 在 sealed 上取得 Schema valid 99.40%、Compile success 99.20%、Intent / route 95.60%；Qwen3.7 Plus API 在相同条件下分别为 57.60%、57.40% 和 52.80%。Lane policy、Lane role exact、Required-lane recall 与 HyDE 也获得 33.80–57.65 个百分点的提升。这个实验说明：**微调确实把通用语言模型转化成了更适合 SoulTuner 检索规划任务的专用模型。**

部署上采用一个模型选择框：当前 4070 电脑继续使用 Qwen3.7 Plus 云端 API；获得 AMD 创空间后，把 **Planner 模块**切到训练后的 35B endpoint。主对话、联网搜索和推荐解释仍可保留 Qwen3.7 Plus，两种 Planner 输出共用严格结构校验、策略守卫和确定性编译器。

本文覆盖：

1. 为什么音乐推荐需要 Graph + Dense，而不是二选一；
2. 如何用公开短证据替代隐藏思维链；
3. AMD MI308X 上 35B MoE Planner 的训练与断点恢复；
4. 如何公平比较 Qwen3.7 Plus API 与微调 35B；
5. 微调给 Planner 结构、路由与 HyDE 带来了多少收益；
6. 如何在 Qwen3.7 Plus 与训练后的 35B 之间一键切换；
7. 可复用的 Gradio 创空间与 Notebook 实践。

配套材料：

- 创空间入口：`deploy/modelscope_space/app.py`；
- 受控策略守卫：`deploy/modelscope_space/planner_guard.py`；
- 训练同款提示词：`deploy/modelscope_space/prompt_v42.py`；
- 35B 候选端点启动脚本：`deploy/modelscope_space/start_35b_endpoint.sh`；
- 可重复执行 Notebook：`notebooks/soultuner_amd_evidence_first_planner.ipynb`。

---

## 1. 项目背景：推荐系统真正需要规划什么？

SoulTuner 的目标不是生成一段看起来合理的推荐文案，而是把自然语言需求编译成可执行的检索计划。后端有三类召回通道：

| 通道 | 擅长解决的问题 | 典型证据 | 不擅长的部分 |
|---|---|---|---|
| Graph | 目录事实、实体关系、离散标签和硬约束 | 歌手、歌曲、年代、语言、流派、情绪标签、场景标签 | 细粒度音色、鼓点、空间感、整体听感 |
| Dense | 声学相似度与难以枚举的主观语义 | 参考歌曲、低音、鼓声、音色、混响、氛围、动态 | 精确事实过滤、最新外部信息 |
| Web | 时效性和本地目录之外的信息 | 最新发行、热榜、新闻、外部资料 | 稳定低延迟的本地召回 |

这里最容易犯的错误是把 Graph 与 Dense 当成互斥分类。实际产品里，它们更像有角色的合作者。

### 1.1 “我心情很差”为什么不是纯 Dense？

“低落、温暖、治愈”可以映射到图谱标签，用来快速缩小候选集；但它们无法完整描述用户期望的声音质感。因此合理计划是：

```json
{
  "lane_policy": {
    "graph": "optional",
    "dense": "required",
    "web": "off"
  },
  "evidence": {
    "reason_codes": [
      "taggable_mood",
      "subjective_affective_goal"
    ],
    "brief_reason": "标签可辅助粗筛，主观听感与相似度由声学检索主导"
  }
}
```

这会被编译成 `dense_primary`：Dense 权重 0.75，Graph 权重 0.25。图谱没有被粗暴关闭，但也不会压过声学相似度。

### 1.2 什么才是接近“纯 Dense”的请求？

下面这类请求主要描述声音本身：

- “我希望 bass 更重一些，鼓声更大一些”；
- “人声靠前、空间感更宽、混响少一点”；
- “找一首和刚刚那首歌听感相似的歌曲”。

如果没有额外的歌手、年代、语言或类型约束，Graph 可以关闭，Dense 必须开启。尤其“刚刚那首歌”不能被错误地写进 `hard.song`——它是相似度锚点，不是要求结果仍然等于那首歌。

### 1.3 什么请求适合 Graph-only？

- “找 90 年代的粤语摇滚”；
- “介绍一下《Bohemian Rhapsody》的创作背景”；
- “只要某位歌手的现场版本”。

这些需求可以被实体和目录字段直接约束，不必强行生成声学 HyDE。

---

## 2. PlannerDecisionV5：把证据、策略和执行拆开

早期 Planner 直接输出 `tool_names=[graph,dense]`。这看似简单，却把三个不同问题混在了一起：

1. 用户说了什么证据？
2. 哪个通道是必须，哪个只是辅助？
3. 具体工具、参数、权重和超时如何配置？

V5 契约改为：

```text
自然语言请求
    ↓
公开、紧凑的 Evidence
    ↓
LanePolicy(required / optional / off)
    ↓
确定性编译器
    ↓
Graph / Dense / Web 工具调用、参数与权重
```

核心字段如下：

```json
{
  "task_mode": "recommendation",
  "dialogue_mode": null,
  "response_mode": "answer",
  "evidence": {
    "decision_phase": "initial",
    "failed_lanes": [],
    "reason_codes": ["acoustic_timbre_or_instrument"],
    "reference_songs": [],
    "brief_reason": "请求描述的是听感或声学特征，应以向量召回为主"
  },
  "lane_policy": {
    "graph": "off",
    "dense": "required",
    "web": "off"
  },
  "hard": {},
  "soft": {},
  "hints": {},
  "metadata": {},
  "acoustic_queries": ["低音更重，鼓声更大"],
  "clarification": null
}
```

### 2.1 为什么 `brief_reason` 只保留 80 字？

训练时加入解释是有价值的，但不应该训练或暴露冗长的隐藏思维链。长推理会带来四个问题：

- 输出 token 和延迟显著增加；
- 模型更容易在解释与最终 JSON 之间自相矛盾；
- 评测变成文字风格比较，而不是行为正确性比较；
- 生产日志可能意外记录不需要的内部推理。

因此 SoulTuner 只监督**可公开、可核验、与执行无关的短理由**。执行只读取结构化证据码和 lane policy，绝不解析自然语言理由。80 个字符足以回答“为什么选择这条通道”，又不会把系统变成思维链生成器。

### 2.2 Evidence Code 比自由文本更重要

典型证据码包括：

- `explicit_entity`：明确歌手或歌曲实体；
- `taggable_genre` / `taggable_mood` / `taggable_scenario`：图谱可表达的标签；
- `acoustic_timbre_or_instrument`：音色或乐器证据；
- `reference_track_similarity`：参考歌曲相似度；
- `freshness_or_external`：需要外部时效信息；
- `unresolved_reference`：指代无法解析，必须澄清；
- `no_retrieval_needed`：普通聊天或产品使用指导。

这些证据可以直接做一致性校验。例如：

- 有参考歌曲或 acoustic query 时，Dense 必须为 `required`；
- 普通聊天与产品使用指导，所有检索通道必须为 `off`；
- `clarify` 模式不得携带检索任务；
- 恢复阶段已经失败的 lane 必须关闭，不能原样重试；
- 参考歌曲不得同时作为结果硬过滤。

---

## 3. 数据工程：不是“多生成一些”就一定更好

Planner 数据最大的风险不是数量少，而是**分布与标注策略错位**。如果训练集中 Dense-only 占比过高，模型就会把所有抽象情绪都路由到 Dense；如果 Graph + Dense 示例过多但没有主次角色，模型又会习惯性地全部开启。

本项目采用三层数据治理：

### 3.1 冻结训练集

- 每条输出必须通过严格 JSON schema；
- 每条数据绑定任务 ID、生成来源、模型接口与时间；
- 数据在训练前冻结 SHA-256，训练中不得改写；
- 训练、regression、sealed 分开管理；
- 私有 sealed 数据不进入公开 Notebook、文章或创空间。

本次正式数据身份为：

| 资产 | SHA-256 |
|---|---|
| train | `4a73fe8e9beb29f1c9983b9b8df01e1cfcadc1b4bb053dd92e246c407185eff6` |
| regression | `df8ed35222db2403473910d060bd27a027b2dd92acbee689296768a0fc8a6b32` |
| sealed | `1383f5f2e0fe388e5ecf1415bd64b0ad69454d984d7906aa056f33ea3653376f` |
| manifest | `4ebaeeadcc843389efdbeb66cdebc2aef6014680f76074a621e6d2d9283c228c` |

### 3.2 Teacher 生成与 Reviewer 审核分离

结构化种子可以用确定性规则迁移；语义含混的样本才值得调用更强 Teacher。批量生成后，再用独立 Reviewer 检查：

- 参考歌曲究竟是 hard filter 还是 acoustic anchor；
- Graph/Dense 是 required、optional 还是 off；
- 信息问答和推荐请求是否混淆；
- 理由是否短、公开且与结构一致；
- 是否存在未知字段、遗漏字段或跨样本污染。

这比给所有历史数据统一调用大模型更省额度，也更可追溯。

### 3.3 regression 与 sealed 的职责不同

`regression` 用来判断新模型是否破坏已知能力，它可以与训练分布相近。`sealed` 才负责检验陌生表达、实体和组合条件下的泛化能力。只看 regression 很容易得出过于乐观的结论。

---

## 4. 在 AMD MI308X 上训练 35B-A3B Planner

### 4.1 硬件与软件边界

- GPU：AMD MI308X；
- 显存：192 GB HBM；
- 训练对象：35B-A3B MoE Planner adapter；
- 总目标：2 epoch；
- 最大序列长度：2048；
- 训练与评测均绑定冻结 manifest；
- checkpoint 同时保存 adapter、trainer state、optimizer、scheduler 与 RNG 状态。

ROCm / PyTorch 具体版本应以运行时环境为准，发布文章时建议插入以下命令的真实输出截图：

```bash
rocminfo | grep -E "Marketing Name|Name:" | head
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
PY
```

> 配图建议 1：MI308X 的 `rocm-smi` 或 `amd-smi monitor` 截图，显示显存、利用率和温度。

### 4.2 为什么必须支持断点恢复？

云实例有固定运行窗口。训练和 500 条生成式评测都可能跨越实例边界，因此恢复策略必须在训练前设计：

- 训练中断：只从最新**完整 checkpoint**恢复；
- 恢复时使用原 `environment.json`；
- `NUM_TRAIN_EPOCHS=2` 表示总目标，而不是再追加两轮；
- 评测中断：绝不续写半成品，创建全新目录，从第 1 条重新运行；
- regression 与 sealed 输出分别保存，避免相互覆盖。

本次训练完成了 2 epoch。实例第一次到期前，regression 412 已完成；sealed 在 51/500 处中断。实例恢复后，我们没有续写这 51 条，而是在新目录重新运行完整 500 条，最终完成一一对应校验、无 thinking 字段检查、评分和归档。

### 4.3 为什么还要保持少量 CPU 活跃？

部分临时实例会把“长时间无前台 CPU 活动”误判为空闲。训练期间额外运行两个 `nice=19` 的低优先级 keepalive，总 CPU 按 23 核折算保持 5% 以上。低优先级保证它不会和数据加载、checkpoint 或推理争抢核心。

这个技巧的原则是：**只维持必要的低优先级活动，不启动额外训练，不改变 epoch，不占用 GPU。**

---

## 5. 微调结果：通用 API 基线与专用 Planner

### 5.1 完整评测结果

| 指标 | Regression 412 | Canonical sealed 500 | 解读 |
|---|---:|---:|---|
| Schema valid | 99.51% | 99.40% | JSON 契约非常稳定 |
| Compile success | 99.51% | 99.20% | 绝大多数输出可编译 |
| Intent / route | 99.03% | 95.60% | 主任务识别较强 |
| Lane policy satisfaction | 91.75% | 86.80% | 必需通道开启、禁止通道关闭的总体满足率 |
| Lane role exact match | 78.88% | 76.20% | Graph/Dense/Web 三个角色逐项完全一致率 |
| Required lane recall | — | 89.98% | sealed 中必需通道的召回率 |
| HyDE present when Dense | 98.88% | 84.44% | Dense 场景生成声学查询的比例 |
| Thinking non-empty | 0 | 0 | 没有输出隐藏思维链 |

### 5.2 Lane policy 与 Lane role exact 的具体含义

`Lane policy satisfaction` 是策略满足率。它重点检查：标准答案标记为 `required` 的通道有没有开启，标记为 `off` 的通道有没有关闭；`optional` 是可以加入但不应成为唯一依据的辅助通道。

`Lane role exact match` 更严格：Graph、Dense、Web 的三个角色必须逐项完全相同。例如标准答案是 `{graph: optional, dense: required, web: off}`，模型输出 `{graph: required, dense: required, web: off}` 时，检索通道都包含在内，但 Graph 的主次角色不同，所以 policy 可能仍满足基本检索要求，而 role exact 不计为命中。

这两个指标的差距说明当前主要提升空间是“通道主次分配”，不是 JSON 输出或任务识别。部署时保留确定性编译器，把 `required / optional / off` 映射为固定执行权重，可以直接利用模型已有能力。

### 5.3 公平对比设置

Qwen3.7 Plus 是 SoulTuner 当前默认使用的 API 模型，因此把它作为部署基线。为了避免提示词、数据和后处理造成不公平差异，本次实验固定以下条件：

- 两个模型使用相同的 canonical sealed 500；
- 请求只包含冻结的 system 与 user 消息，标准答案不发送给模型；
- 使用相同的 PlannerDecisionV5 输出契约；
- temperature=0、thinking 关闭；
- 不使用自动补字段、规则修复或人工修订；
- 使用同一个确定性 scorer，缺失或非法输出直接计错；
- Qwen3.7 Plus 完成 500/500，请求错误为 0，运行时模型身份回读为 `qwen3.7-plus`。

### 5.4 核心对比结果

| Canonical sealed 500 | 微调 35B | Qwen3.7 Plus API | 微调收益 |
|---|---:|---:|---:|
| Schema valid | 99.40% | 57.60% | +41.80pp |
| Compilable | 99.20% | 57.40% | +41.80pp |
| Intent / route | 95.60% | 52.80% | +42.80pp |
| Lane policy satisfaction | 86.80% | 48.40% | +38.40pp |
| Lane role exact match | 76.20% | 42.40% | +33.80pp |
| Required lane recall | 89.98% | 32.33% | +57.65pp |
| HyDE present when Dense | 84.44% | 48.89% | +35.55pp |
| Thinking non-empty | 0 | 0 | 持平 |

提升不是只来自“JSON 更整齐”。Intent / route 提升 42.80 个百分点，Required-lane recall 提升 57.65 个百分点，说明模型同时学到了任务类型、Graph/Dense/Web 召回职责和声学查询生成。结构正确率与执行语义同步提高，才构成一次有效的 Planner 微调。

### 5.5 应该如何解释“微调成功”？

这个结论严格限定在 SoulTuner PlannerDecisionV5 任务：微调 35B 更适合承担项目中的**检索规划模块**。它不等于 35B 在所有通用问答、联网搜索、长文本写作或知识覆盖上全面超过 Qwen3.7 Plus。

因此实际产品采用模块化分工：

- 微调 35B：意图、证据、Lane 角色、HyDE 与可执行计划；
- Qwen3.7 Plus：通用对话、联网信息、推荐解释与没有 35B 资源时的云端回退；
- Guard + Compiler：两种 Planner 的共同执行边界。

---

## 6. 最快部署：模型下拉框 → Guard → Compiler

### 6.1 线上架构

```text
用户请求
   │
   ├── 模型下拉框：Qwen3.7 Plus / SoulTuner 35B / 无模型演示
   │        生成候选 V5 JSON
   │
   ▼
Strict Schema Validator
   │  非 JSON / 未知字段 / 超时 ───────────┐
   ▼                                      │
Policy Guard                              │
   │  必需 lane 遗漏 / 对话冲突 ──────────┤
   ▼                                      │
Deterministic Compiler                    │
   │                                      │
   ▼                                      ▼
Graph / Dense / Web 执行             Safe Fallback Plan
   │                                      │
   └──────────────────┬───────────────────┘
                      ▼
                  结果融合与回答
```

### 6.2 守卫检查什么？

1. `task_mode`、`dialogue_mode`、`response_mode` 必须合法；
2. lane 只能是 `required / optional / off`；
3. Dense 不允许 `optional`；
4. 声学特征或参考歌曲出现时，Dense 必须 `required`；
5. 信息问答必须查询 Graph，不允许 Dense；
6. 普通聊天与产品指导不允许任何检索；
7. 上下文指代无法解析时必须澄清；
8. `brief_reason` 必填且不超过 80 字；
9. 模型候选与确定性最低安全分类冲突时，拒绝候选；
10. 候选端点超时或失败时，自动降级。

### 6.3 确定性权重

| Graph | Dense | Profile | Graph 权重 | Dense 权重 |
|---|---|---|---:|---:|
| required | off | graph_only | 1.00 | 0.00 |
| required | optional | graph_primary | 0.75 | 0.25 |
| required | required | balanced_hybrid | 0.50 | 0.50 |
| optional | required | dense_primary | 0.25 | 0.75 |
| off | required | dense_only | 0.00 | 1.00 |

权重不由模型自由生成，避免模型输出看似合理但不可控的浮点数。

### 6.4 4070 与 AMD 创空间怎么分工？

RTX 4070 不需要承载 35B。当前电脑只运行 Gradio、守卫和推荐应用，Planner 选 Qwen3.7 Plus 云端 API；35B 基座与 adapter 留在 AMD MI308X 创空间或远程服务器。获得资源后，只需配置 endpoint 并在下拉框切换模型：

```text
SOULTUNER_MODEL_PROFILE=soultuner-v4.2-35b
SOULTUNER_PLANNER_ENDPOINT=https://<host>/v1/chat/completions
SOULTUNER_PLANNER_MODEL=soultuner-planner-v4.2-35b
```

没有 35B 端点时选择 Qwen3.7 Plus；没有 API Key 时选择安全演示。三种档位共用相同页面和输出格式。

---

## 7. 创空间部署与 ROCm 运行

项目已提供独立 Gradio 包：

```text
deploy/modelscope_space/
├── app.py
├── model_profiles.py
├── .env.example
├── planner_guard.py
├── prompt_v42.py
├── start_35b_endpoint.sh
├── requirements.txt
└── README.md
```

本地启动：

```powershell
cd deploy/modelscope_space
python -m pip install -r requirements.txt
$env:DASHSCOPE_API_KEY="你的 Key"
$env:SOULTUNER_MODEL_PROFILE="qwen3.7-plus"
python app.py
```

创空间使用 `app.py` 作为入口，端口 7860。没有模型权重时仍能展示 Evidence、LanePolicy、编译权重和降级行为；批准 AMD MI308X 资源后，可通过 Secret 配置 35B HTTPS 端点：

```text
SOULTUNER_PLANNER_ENDPOINT=https://<your-endpoint>/planner/v5
SOULTUNER_PLANNER_TOKEN=<secret>
```

如果直接使用训练模型的 OpenAI-compatible 服务，端点为 `https://<host>/v1/chat/completions`。Token 只放创空间 Secret，不进入代码仓库。超时由 `SOULTUNER_PLANNER_TIMEOUT` 配置；35B 冷启动阶段建议 180 秒，出错即降级。

本次在 AMD MI308X 上恢复 `checkpoint-450` 后完成了真实部署验证：服务身份回读为 `soultuner-planner-v4.2-35b`，公开请求“我今天心情很差，想听一些温暖、治愈但不要太吵的歌”经热启动端到端耗时 24.56 秒，候选通过 Policy Guard，最终编译为 Graph 0.25 / Dense 0.75，非空 thinking 为 0。该结果用于证明部署链路可运行，不包含私有 sealed 样本。

> 配图建议 2：创空间主页，左侧输入“我心情很差”，右侧显示 `dense_primary` 与 Graph 0.25 / Dense 0.75。
> 配图建议 3：人为构造一个遗漏 Dense 的候选，展示“候选被拒绝，已回退到确定性安全计划”。

---

## 8. Notebook 实践设计

配套 Gallery Notebook 使用公开合成样本，可以在没有私有模型与数据的环境中完整执行：

1. 检测 ROCm / AMD GPU 环境；
2. 定义 Evidence-first 契约；
3. 演示 Graph-only、Dense-only、Dense-primary 和 clarification；
4. 对模型候选做 fail-closed 守卫；
5. 绘制 regression / sealed 指标差距；
6. 可选调用真实 Planner endpoint；
7. 打印部署检查清单。

Notebook 默认 smoke mode 全 Cell 可运行。真实端点只是可选步骤，不配置时会明确跳过，不会报错。

---

## 9. 下一步：不继续盲目增加训练量

下一轮训练应由线上错误分布驱动，而不是平均扩充所有题型。优先收集：

1. Graph optional + Dense required 的情绪/听感组合；
2. Graph required + Dense off 的信息问答；
3. resolved reference 与 unresolved reference 对照样本；
4. 同时含图谱标签与主观听感、但主次不同的成对请求；
5. 生产候选被 Policy Guard 拒绝的真实匿名样本。

推荐用定向 hard-negative 和对比样本：只改变一个语义因素，让正确 lane 角色随之改变。例如：

```text
A: “找温暖治愈的歌”                → graph optional, dense required
B: “找标签为治愈的华语歌曲”          → graph required, dense off
C: “找低音更重、鼓声更大的歌”        → graph off, dense required
D: “介绍这首歌的发行年份”            → dialogue information, graph required
```

这类数据比再生成一批同分布的普通推荐请求更可能提升 sealed 指标。

---

## 10. 结论

这次实践的核心经验是：**领域微调的价值，必须在真实项目契约和真实替代基线上验证。**

- 相比项目当前使用的 Qwen3.7 Plus API，微调 35B 在 V5 Planner 核心指标上提升 33.80–57.65 个百分点；
- 35B 模型可以学会非常稳定的结构化输出；
- regression 很高不代表 sealed 泛化已经可靠；
- Graph 与 Dense 应表达角色，而不是简单开关；
- 简短公开证据比冗长思维链更适合监督与审计；
- 部署端用统一 Guard 与 Compiler 承接不同模型，切换模型不改业务代码；
- AMD MI308X 提供了完成 35B MoE 训练、断点恢复和大规模评测所需的显存空间，而 ROCm 环境让训练与 PyTorch 生态保持一致。

SoulTuner 当前已经具备受控部署条件：Qwen3.7 Plus 保留通用能力和云端可用性，微调 35B 负责更准确的领域规划，策略守卫负责把错误限制在执行之前。训练结果证明了专用 Planner 的收益；获得 AMD 创空间后即可把 Planner 默认档位切到 35B。

---

## 附录 A：可复现实验身份

| 项目 | 值 |
|---|---|
| 训练运行 ID | `planner-v4.2-35b-2ep-20260810T164032Z` |
| 代码提交 | `7c543bb6f66ffba8fcc25dfd74ee157a1e424c55` |
| Manifest SHA | `4ebaeeadcc843389efdbeb66cdebc2aef6014680f76074a621e6d2d9283c228c` |
| Best checkpoint | `checkpoint-450` |
| Last checkpoint | `checkpoint-984` |
| 训练目标 | 2 epoch（总目标） |
| Regression | 412 / 412 |
| Canonical sealed | 500 / 500 |
| Thinking non-empty | 0 |

## 附录 B：发布前检查表

- [ ] 标题或正文明确 AMD MI308X / ROCm；
- [ ] 携带 `AMD GPU激励计划` 标签；
- [ ] 加入 AMD GPU 监控截图；
- [ ] 加入创空间运行截图；
- [ ] 代码仓库与 Notebook 链接可访问；
- [ ] 不上传私有 train / regression / sealed 原文；
- [ ] 不上传 API Token、Cookie、SSH Key；
- [ ] 明确 MuQ-MuLan 权重的 CC-BY-NC 许可边界，商业部署前替换或获得授权；
- [ ] 不把情绪音乐推荐描述成医疗诊断或治疗；
- [ ] 公开指标与归档记录一致。

## 参考资料

1. ModelScope × AMD 开发者激励计划：<https://modelscope.cn/events/247>
2. ModelScope Hub 官方客户端：<https://github.com/modelscope/modelscope_hub>
3. AMD ROCm 文档：<https://rocm.docs.amd.com/>
4. SoulTuner 项目仓库中的 `deploy/modelscope_space` 与配套 Gallery Notebook。
