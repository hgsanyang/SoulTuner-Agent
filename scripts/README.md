# scripts/

运维与部署脚本。

| 文件 | 职责 |
|------|------|
| `download_models.py` | 自动下载 M2D-CLAP / BERT / OMAR-RQ 模型权重（检测已有文件自动跳过） |
| `start_sglang.sh` | WSL 环境下启动 SGLang 推理引擎（本地 Qwen3-4B 部署） |
| `start_vllm.sh` | WSL 环境下启动 vLLM 推理引擎 |
| `start_vllm.bat` | Windows 环境下启动 vLLM（批处理脚本） |
| `reset_user_data.py` | 重置用户数据（清除 Neo4j 中的用户行为与偏好记录） |
