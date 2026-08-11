# SoulTuner 最快发布、AMD 激励与参赛执行计划

更新时间：2026-08-11（Asia/Shanghai）

## 结论先行

停止追加训练，直接完成可访问部署。今天的发布目标分成两层：

1. **可访问版本**：现有稳定 Planner / API 主路 + 确定性安全计划，先保证应用可用；
2. **35B 增强版本**：获得 AMD 创空间后，在同一个模型下拉框切到训练后的 35B endpoint。

当前 4070 不加载 35B 权重：只运行应用和守卫，Planner 使用 Qwen3.7 Plus 云端 API。两种模型共用 schema、lane policy 和超时守卫。

## 一、当前交付物

| 交付物 | 路径 | 状态 |
|---|---|---|
| 原创技术文章 | `docs/modelscope_amd/SOULTUNER_AMD_TECH_ARTICLE.md` | 完整初稿，待补截图/链接 |
| 灵感流 Notebook | `notebooks/soultuner_amd_evidence_first_planner.ipynb` | 自包含，7 个代码 Cell 已逐项执行 |
| 创空间应用 | `deploy/modelscope_space/app.py` | Gradio 入口已完成 |
| 模型档位 | `deploy/modelscope_space/model_profiles.py` | Qwen / 35B / safe 一键切换 |
| 策略守卫 | `deploy/modelscope_space/planner_guard.py` | 单测通过 |
| 训练同款 prompt | `deploy/modelscope_space/prompt_v42.py` | 已冻结 |
| 35B 推理启动脚本 | `deploy/modelscope_space/start_35b_endpoint.sh` | 使用官方 `swift deploy` |

## 二、48 小时发布节奏

### T+0 至 T+2 小时：先发布不依赖 72 GB 权重的创空间

1. 注册/确认 AMD Developer 账号；
2. 申请加入 ModelScope `AMD_Dev` 组织；
3. 创建公开 Gradio 创空间；
4. 上传 `app.py`、`model_profiles.py`、`planner_guard.py`、`prompt_v42.py`、`requirements.txt`；
5. 在 CPU/默认资源上验证六个公开示例；
6. 公开页面明确显示“确定性安全演示”或“35B 候选＋守卫”，不伪装运行模式。

验收标准：

- 页面可从无登录浏览器打开；
- “心情很差”得到 `dense_primary`；
- “低音更重”得到 `dense_only`；
- “怎么导入歌单”不触发检索；
- 未解析“刚刚那首”必须澄清；
- 页面不暴露 Token、路径、私有数据。

### T+2 至 T+6 小时：启动 35B OpenAI 兼容端点

在 AMD 实例上用 `checkpoint-450`：

```bash
export SOULTUNER_BASE_MODEL=/path/to/Qwen3.6-35B-A3B
export SOULTUNER_ADAPTER=/path/to/checkpoint-450
export SOULTUNER_SERVE_API_KEY='<random-secret>'
bash deploy/modelscope_space/start_35b_endpoint.sh
```

先使用与正式评测相同的原生 ms-swift 路径。只有在单独验证 ROCm vLLM + LoRA 输出一致性后，才切换 vLLM。

创空间配置：

```text
SOULTUNER_PLANNER_ENDPOINT=https://<endpoint>/v1/chat/completions
SOULTUNER_PLANNER_MODEL=soultuner-planner-v4.2-35b
SOULTUNER_PLANNER_PROTOCOL=openai
SOULTUNER_PLANNER_TOKEN=<secret>
```

验收标准：10 个公开合成请求连续成功；人为制造一次非法候选，页面必须显示被拒绝并降级；关闭端点后应用仍可工作。

### T+6 至 T+12 小时：发布 Notebook

1. 在 Gallery 创建公开 Notebook；
2. 上传 `.ipynb`；
3. 选择 AMD GPU 环境执行全部 Cell；
4. 保存输出，加入 MI308X / ROCm 截图；
5. 标签使用 `AMD GPU激励计划`；
6. 描述中链接技术文章和创空间。

审核关键点：全部 Cell 无报错、中文/英文注释完整、依赖说明完整、与 AMD GPU / ROCm 直接相关。

### T+12 至 T+24 小时：发布技术文章

文章采用现有长文，不再写泛泛教程。补充：

- AMD GPU 监控截图；
- 创空间 `dense_primary` 截图；
- Qwen3.7 Plus / SoulTuner 35B 模型下拉框截图；
- Notebook、代码仓库与创空间公开链接。

文章标签使用 `AMD GPU激励计划`。正文展示训练指标、Graph/Dense 设计、一键部署方式和当前精细 Lane 角色的优化重点。

### T+1 至 T+2 天：完善主项目展示

- 当前 Next.js + FastAPI 产品继续走稳定主路；
- 在设置页显示 Planner 模式：Qwen3.7 Plus / SoulTuner 35B / safe；
- 记录候选接受率、拒绝原因与降级率，不记录私有思维链；
- 对外 Demo 不自动播放或再分发未授权音乐文件；
- 将 35B 切换设计成 feature flag，随时一键回滚。

## 三、AMD × ModelScope 激励收益路径

活动官方页面显示截止时间为 **2026-12-31 15:59**，三条路径可独立参与并累计：

| 路径 | 单项奖励 | 上限 | SoulTuner 对应作品 |
|---|---:|---:|---|
| 研习社原创技术文章 | 25 GPU 小时/篇 | 50 小时 | 本项目训练、评测和受控部署长文 |
| Gallery Notebook | 50 GPU 小时/篇 | 150 小时 | Evidence-first Planner 实践 |
| 创空间应用 | 按需 700 小时或更多 | 最高资源路径 | SoulTuner Planner Gradio 应用 |

前置的 AMD Developer 注册还可获得一次性 100 小时。创空间优先分配通常要求至少已有一篇审核通过的文章或高质量 Notebook，因此执行顺序是：**创空间先上线可见版本，同时提交 Notebook 和文章，审核通过后补 AMD 专属资源申请。**

组织申请入口：<https://www.modelscope.cn/organization/AMD_Dev>

## 四、当前仍可报名的赛事

### P0：Hack for Humanity Summer 2026（直接适配，优先报名）

- 时间：2026-08-07 至 2026-09-04；
- 形式：全球线上，13 岁以上，个人或 2–4 人团队；
- 要求：可运行软件、GitHub 源码、最长 4 分钟视频；
- 适配方向：`SoulTuner Calm — privacy-first mood-aware music companion`；
- 目标赛道：Mental Health、Best Use of AI/ML、Responsible AI、Best Design；
- 官方页：<https://hack-for-humanity-summer-26.devpost.com/>。

需要新增而不是冒充已有成果：

1. Well-being 模式与清晰的非医疗声明；
2. 情绪输入只用于当次推荐，默认不持久化；
3. 本地/匿名偏好、删除数据入口；
4. 危机/自伤表达不做“音乐治疗”，展示专业求助提示；
5. 单独记录 2026-08-07 之后为赛事新增的功能和提交。

注意：不要宣称诊断、治疗抑郁或焦虑。产品定位是音乐陪伴与情绪自我调节辅助。

### P1：AI Builders Hackathon（即将开始，仅在你满足“学生”资格时）

- 时间：2026-08-21 至 2026-09-15；
- 页面当前显示总奖金 33,900 美元；
- 交付：工作产品、公开源码、5 分钟 Demo、最多 10 页 deck；
- 技术/产品匹配度很高；
- 但 Devpost 资格栏明确写 **Students only**，页面正文“everyone welcome”与此冲突，应以正式 Rules 为准；
- 官方页：<https://ai-builders-hackathon-2026.devpost.com/>。

如果不是在读学生，不投入制作时间；如果是，SoulTuner 当前版本可以直接作为主体，重点展示真实用户价值、可访问 Demo、受控 AI 与长期记忆隐私。

### P1：VoltHacks 2026（同样仅限学生）

- 截止：2026-09-05 17:00 EDT；
- 全球线上，13+，Students only；
- 主题：Hardware / IoT / AI；
- 奖池标称 32,785 美元；评委中包含 Sony Music Entertainment 数据科学人员；
- 官方页：<https://volthacks.devpost.com/>。

当前项目缺少硬件叙事，因此只有在学生资格成立、并能增加本地 AMD 推理/边缘音乐场景时才投入，优先级低于 Hack for Humanity。

### P2：SUFE Sequential Recommendation Kaggle（算法练习，不是产品赛）

- 当前仍开放，页面显示约一个月；
- 任务：Amazon Beauty 用户序列 next-item 推荐；
- 指标：mean NDCG@10；
- 本来是 SUFE Deep Learning 2026 Spring 课程项目；只有 Kudos，无 Kaggle points/medals 和奖金；
- 官方页：<https://www.kaggle.com/competitions/sufe-deep-learning-2026-spring-sequential-recommendat/>。

它不能直接提交 SoulTuner 应用，只能迁移本项目的序列偏好建模方法并提交 CSV。适合补齐推荐算法履历，不应挤占当前发布工作。

### Watchlist：TechEx Amsterdam Hackathon

- lablab 官方首页列出的未来活动：线上 Build 2026-10-16 至 10-19，现场 10-19 至 10-20；
- 适合在创空间和比赛 Demo 完成后报名；
- 当前公开赛题细则仍少，先关注，不提前做定制开发；
- 官方入口：<https://lablab.ai/>。

## 五、明确排除，避免再次浪费时间

- **Agentic Cinema**：官方页明确把 China 列为不允许参赛地区；
- **BenchFlow Agent Skill Lift**：最终提交截止 2026-07-08，已经结束；
- **AMD 黑客马拉松、GeoAI Challenge、数龙杯等**：已过报名/提交日期；
- **CUHK-X Large Model Track**：仍开放到 2026-09-15，但任务是多模态人体活动 VQA，当前音乐项目不能直接提交；
- **Arm Create**：2026-08-14 截止且要求 Arm 优化，时间和技术栈都不匹配；
- **SUFE 推荐赛**：只能算算法适配，不能把当前产品链接当作有效提交。

## 六、参赛与发布共用素材

只做一套资产，复用到 ModelScope 与 Devpost：

1. 90 秒产品短片：情绪输入 → 证据 → Graph/Dense 路由 → 歌曲结果；
2. 4 分钟技术 Demo：再加入 MI308X、35B、Policy Guard、fallback；
3. 一张架构图；
4. 一张真实评测表；
5. 一张隐私/Responsible AI 边界图；
6. 公开仓库 README；
7. 在线创空间链接；
8. 文章与 Notebook 互链。

禁止使用无授权歌曲作为公开视频背景音乐。演示可以显示歌曲元数据和短预览链接，但不要把受版权保护音频提交进仓库。

## 七、上线验收门槛

这是受控 MVP 的上线门，而不是 35B 全替换门：

- [ ] 创空间公开可访问；
- [ ] 六个代表性请求行为正确；
- [ ] 候选端点断开时自动降级；
- [ ] 非 JSON、超时、缺少 Dense 三种失败均被守卫拦截；
- [ ] 线上主路不依赖 35B 存活；
- [ ] 不展示私有数据和 sealed 原文；
- [ ] Secret 不进仓库和日志；
- [ ] 明确 MuQ-MuLan 权重 CC-BY-NC，商业化前更换或获授权；
- [ ] Well-being 场景有非医疗声明与危机安全边界；
- [ ] 有一键关闭 35B candidate 的 feature flag。

完成以上条件后即可部署运行；不需要等待新一轮训练。
