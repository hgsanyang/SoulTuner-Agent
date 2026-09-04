# 参与 SoulTuner Agent

感谢你愿意改进项目。这个仓库采用轻量的个人开源协作流程：任何人可以提出建议或提交修改，但所有变更都需要维护者审查后才会进入 `main`。

## 提建议或报告问题

- 功能建议使用 GitHub 的“功能建议”Issue 模板。
- 可复现的异常使用“问题报告”模板，并附最小复现步骤、运行环境和脱敏日志。
- 安全漏洞、令牌、数据库凭据和未公开利用细节不要写入公开 Issue，请按 [SECURITY.md](SECURITY.md) 私下报告。
- 大范围架构改动建议先开 Issue 对齐目标，避免投入大量工作后方向不一致。

## 提交代码

1. Fork 仓库，从最新 `main` 创建单一目的的分支。
2. 保持改动聚焦，不提交模型权重、音频、数据集缓存、密钥或 sealed 评测材料。
3. 为行为变化补充单元测试；涉及前端时同时检查 lint 和 production build。
4. 提交 Pull Request，说明动机、主要变化、验证结果以及兼容性影响。
5. 根据审查意见更新。PR 是修改提案，不代表自动获得合并权限。

常用验证命令：

```bash
python -m pip install -r requirements-ci.txt
ruff check .
python -m pytest tests/unit/ -q

cd web
npm ci --no-audit --no-fund
npm run lint
npm run build
```

## 项目边界

- 推荐意图由 Planner 与结构化契约理解，不用固定触发词替代自然语言理解。
- GPU 检索以 MuQ 中文语义召回和 OMAR 声学重排为主；纯 CPU 档使用 M2D-CLAP。
- 联网结果只能补充已有候选，不能覆盖本地/图谱召回。
- 新音频必须具备可核验来源、逐曲许可证与归属信息。
- 不得提交训练私有数据、教师/API 原始输出、用户行为数据或 sealed 评测答案。

提交贡献即表示你有权提供相关代码和素材，并同意按仓库的 MIT 许可证发布代码贡献。第三方音频、模型与数据仍遵循各自许可证。
