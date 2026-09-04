# Changelog

本文件记录对使用者和贡献者有意义的变化。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，项目版本遵循语义化版本原则。

## [Unreleased]

### Added

- ModelScope AMD MI308X 公共演示、35B Planner/自然对话双角色端点与开放音频曲库。
- MuQ 中文语义召回与 OMAR 声学二阶段重排；纯 CPU 档保留 M2D-CLAP。
- 贡献指南、安全报告流程、Issue/PR 模板与依赖更新检查。

### Changed

- 推荐曲目先返回，35B 推荐话术随后流式补充，减少首屏等待。
- 后续自然语言需求统一交给 Planner 判断，移除固定推荐触发词依赖。
- 联网搜索成为补充候选，不再覆盖已有本地或图谱候选。

### Fixed

- 后续推荐意图不刷新歌单的问题。
- Aura/Neo4j 短暂断线后的自动重连与安全本地降级。
- 公开创空间启动时重复下载已持久化曲库和模型的问题。
