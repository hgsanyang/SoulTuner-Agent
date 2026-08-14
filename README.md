# 🎵 SoulTuner Agent

<p align="center">
  <img src="assets/logo.png" alt="logo" width="200" />
</p>

<p align="center">
  <strong>A natural-language music recommendation agent</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/LangGraph-Agent_Framework-orange?logo=langchain" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Neo4j-Graph_Database-008CC1?logo=neo4j" alt="Neo4j" />
  <img src="https://img.shields.io/badge/Next.js_14-Frontend-black?logo=next.js" alt="Next.js" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker" alt="Docker" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
  <br/>
  <img src="https://github.com/hgsanyang/SoulTuner-Agent/actions/workflows/ci.yml/badge.svg" alt="CI" />
  <img src="https://img.shields.io/badge/tests-1200+_passed-brightgreen?logo=pytest" alt="Tests" />
  <img src="https://img.shields.io/badge/code_style-ruff-261230?logo=ruff" alt="Ruff" />
</p>

<p align="center">
  <a href="README.ch.md">中文</a> | <a href="README.md">English</a>
</p>

## 🎯 What it is

SoulTuner is an open-source music recommendation agent. Describe a mood, scene, sound, artist, or a song you want to avoid in one ordinary sentence. SoulTuner turns that request into a search plan, looks through the music library, and explains why each result fits.

- 🗣️ **Just say it** — "I'm feeling really down today, I just want some quiet time alone." No need to pick genres or keywords first.
- 🔎 **Searches in more than one way** — combines known music facts with sound similarity instead of relying on a single keyword search.
- 🧠 **Gets to know you** — likes, saves, skips, and preferences you explicitly share can shape later ranking without replacing your current request.
- 🌐 **Goes online when your library falls short** — optional web discovery can supplement the local catalogue; it can be turned off at any time.
- ♻️ **Grows a lawful music library** — approved audio can be checked, described, converted into searchable representations, reviewed, and then added to the catalogue.
- 🧪 **Daily and developer modes** — anything you do in developer mode is stored separately: it never feeds personalisation and never reaches the training set.

> 📖 Full feature and interaction details: [Feature_Walkthrough.md](Feature_Walkthrough.md)

---

## 🖼️ Preview

<p align="center">
  <a href="https://www.bilibili.com/video/BV11dQLBDEeF/">
    <img src="https://img.shields.io/badge/▶_Demo_Video_—_Watch_on_Bilibili_|_BV11dQLBDEeF-00A1D6?style=for-the-badge&logo=bilibili&logoColor=white&labelColor=FB7299" alt="Demo Video" />
  </a>
</p>

### 🏠 Home · 💬 Chat · 🎵 Recommendations · 🎧 Player

<table>
  <tr>
    <td><img src="assets/首页.png" alt="Home" /></td>
    <td><img src="assets/对话页面.png" alt="Chat" /></td>
  </tr>
  <tr>
    <td><img src="assets/音乐推荐.png" alt="Recommendations" /></td>
    <td><img src="assets/播放页1.png" alt="Player" /></td>
  </tr>
</table>

---

## ✨ How it works

1. **Understand the request** — the Planner separates hard requirements (for example, artist or language) from softer preferences such as mood and atmosphere.
2. **Find candidates** — graph search finds music with the right facts and relationships; vector search finds music that sounds or feels similar. Results are merged and diversified.
3. **Learn from feedback** — likes, saves, skips, and session context are recorded as traceable events. Useful preferences can be recalled in later conversations; stale or unrelated context stays out.
4. **Improve the catalogue** — the data pipeline checks approved audio, adds tags and background information, creates music embeddings, and imports the reviewed result into the searchable library.

The language model plans *how to search*; deterministic application code validates that plan before any retrieval tool runs. The model does not invent a song list and bypass the catalogue.

### A Planner built for this project

The default setup can use the Qwen3.7 Plus API. SoulTuner also includes an optional 35B Planner trained specifically for its retrieval contract. On a held-out 500-request planning evaluation, the trained Planner produced valid structured decisions for **99.4%** of requests and selected the correct intent and retrieval route for **95.6%**. These figures measure planning behaviour, not subjective music quality.

Both Planner options use the same retrieval, memory, ranking, and frontend code. Switching models therefore does not require rewriting the recommendation system.

---

## 🚀 Quick start

```powershell
cd <your project directory>
Copy-Item .env.example .env
notepad .env
```

Fill in at least these (the default setup uses DashScope / Qwen):

```env
MAIN_LLM_PROVIDER=dashscope
MODEL_NAME=qwen3.7-plus
DASHSCOPE_API_KEY=your DashScope key
NEO4J_PASSWORD=your Neo4j password
MUSIC_DATA_PATH=../data
```

Then start it and open `http://localhost:3003`:

```powershell
.\soultuner.ps1 up gpu
```

Without an NVIDIA GPU, use `.\soultuner.ps1 up cpu`.

To use another provider (SiliconFlow, Google, Volcengine, or local SGLang / vLLM / Ollama), change `MAIN_LLM_PROVIDER` and `MODEL_NAME` and supply the matching key — or adjust it from **System Settings** in the UI after startup.

### Choose a Planner

SoulTuner accepts an API model or a self-hosted OpenAI-compatible endpoint. The large model can stay on a GPU server while the rest of the application runs on an ordinary computer.

| Option | Best for | What you need |
|---|---|---|
| Qwen3.7 Plus API | the easiest first run | an API key; no large local GPU |
| SoulTuner V4.2 35B | project-specific planning and private hosting | a high-memory inference server |
| Safe demo | UI and retrieval demonstration | CPU only; no external model call |

See [the self-hosting package](deploy/self_hosted_35b) for the model switch, integrity checks, server startup, and benchmark tools. The main Docker deployment supports CPU and NVIDIA CUDA; [AMD ROCm deployment](docs/AMD_ROCM_DEPLOYMENT.md) is available as an overlay without changing the application code.

<details>
<summary>Other common commands</summary>

| Command | Purpose |
|---|---|
| `.\soultuner.ps1 doctor` | Check that the services are healthy |
| `.\soultuner.ps1 down` | Stop all containers |
| `.\soultuner.ps1 logs` | Tail service logs |
| `.\soultuner.ps1 test` | Run the unit tests |
| `.\soultuner.ps1 ingest gpu` | Process the pending-ingest queue on the GPU worker |
| `python scripts/dev/start_backend.py` | Backend only, for local debugging |

</details>

---

## 🏗️ Architecture

One recommendation request travels this path:

```
your sentence
     │
     ▼
┌──────────────────────────────────────────────────┐
│  Agent (LangGraph)                                │
│  recall memory → LLM plan → route by intent       │
│  find songs / chat / acquire / clarify            │
└──────────────────────┬───────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────┐
│  Retrieval and catalog expansion                  │
│  graph ＋ MuQ vector → fuse → optional web fill   │
└──────────────────────┬───────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────┐
│  Storage: Neo4j (graph + vectors + behaviour)     │
│           SQLite (memory ledger + feedback events)│
└──────────────────────┬───────────────────────────┘
                       ▼
        SSE streaming → frontend (Next.js)
                       │
                       ▼
        your feedback ─┘  recorded, and updates your taste profile
```

### Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14 + React 18 |
| Backend | FastAPI + SSE streaming |
| Agent | LangGraph StateGraph |
| Graph database | Neo4j 5.x (relations + native vector index) |
| Text-to-music | MuQ-MuLan (M2D-CLAP fallback) |
| LLM | `dashscope / qwen3.7-plus` by default, provider swappable |
| Long-term memory | Local SQLite ledger + Neo4j hot path |
| Ranking | Multi-source fusion → rerank → diversity |
| Deployment | Docker Compose (CPU / GPU entrypoints) |

> 📖 How to run the recommendation-quality and alignment evaluations: [tests/eval/README.md](tests/eval/README.md)

---

## 📁 Layout

```
agent/       LangGraph workflow and intent routing
retrieval/   hybrid retrieval, fusion & ranking, audio encoders, context pipeline
tools/       graph search / text-to-music / web discovery / song acquisition
services/    memory gateway, feedback events, ranking policy, service clients
schemas/     Pydantic contracts (state, query plan, feedback events)
llms/        provider registry and prompts
api/         FastAPI layer
data/        data pipeline and planner distillation harness
web/         Next.js frontend
tests/       unit tests + outcome-oriented evaluation
```

The planner can be distilled into a local student model. The public repository
ships the reproducible harness, while private training data stays outside Git;
see [data/sft/README.md](data/sft/README.md).

---

## ⚙️ Configuration

| Variable | Purpose |
|---|---|
| `DASHSCOPE_API_KEY` | Key for the default model (use your provider's key if you switch) |
| `NEO4J_PASSWORD` | Local Neo4j password |
| `MUSIC_DATA_PATH` | Where audio, caches, the ingest queue and feedback logs live |
| `MUSIC_WEB_SEARCH_ENABLED` | Whether web supplementation is allowed |
| `ADMIN_API_KEY` | Optional. Set it and delete / settings / rebuild require the key |

See `.env.example` for the advanced options; normal use needs none of them.

It listens on `127.0.0.1` only. For remote access use a VPN or SSH tunnel.

---

## 🙏 Acknowledgements

The initial architecture came from [imagist13/Muisc-Research](https://github.com/imagist13/Muisc-Research) and has since been substantially rebuilt and extended.

| Project | Used for |
|---|---|
| [OpenMuQ/MuQ](https://github.com/OpenMuQ/MuQ) | MuQ-MuLan, the primary text-to-music model (CC-BY-NC 4.0) |
| [nttcslab/m2d](https://github.com/nttcslab/m2d) | M2D-CLAP fallback and auxiliary semantic model |
| [MTG/omar-rq](https://github.com/MTG/omar-rq) | OMAR-RQ audio representation model |
| [aexy-io/graphzep](https://github.com/aexy-io/graphzep) | legacy memory adapter (optional, non-default) |

---

## 📚 References

- Palumbo, E. et al. (2025). *You Say Search, I Say Recs.* RecSys 2025.
- Zhu, H. et al. (2025). *MuQ / MuQ-MuLan: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization.* [arXiv:2501.01108](https://arxiv.org/abs/2501.01108)
- Niizumi, D. et al. (2025). *M2D-CLAP: Exploring General-purpose Audio-Language Representations Beyond CLAP.* IEEE Access. [arXiv:2503.22104](https://arxiv.org/abs/2503.22104)
- Alonso-Jiménez, P. et al. (2025). *OMAR-RQ: Open Music Audio Representation Model.* ACM MM 2025. [arXiv:2507.03482](https://arxiv.org/abs/2507.03482)
- Gao, L. et al. (2023). *Precise Zero-Shot Dense Retrieval without Relevance Labels.* ACL 2023.
- Xu, W. et al. (2025). *A-MEM: Agentic Memory for LLM Agents.* [arXiv:2502.12110](https://arxiv.org/abs/2502.12110)
- Wang, Y. et al. (2023). *RecMind: Large Language Model Powered Agent for Recommendation.* [arXiv:2308.14296](https://arxiv.org/abs/2308.14296)
- Wu, D. et al. (2025). *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.* ICLR 2025.
- Manco, I. et al. (2023). *The Song Describer Dataset.* [arXiv:2311.10057](https://arxiv.org/abs/2311.10057)
- Rasmussen, P. et al. (2025). *Zep: A Temporal Knowledge Graph Architecture for Agent Memory.*

---

## 📄 License

- **SoulTuner source code:** MIT (see [LICENSE](LICENSE)).
- **MuQ-MuLan model weights:** CC-BY-NC 4.0 — **non-commercial only**. The default setup downloads these weights, so *using the default configuration commercially requires replacing them or obtaining a separate licence for the restricted models.* M2D-CLAP and OMAR-RQ carry their own upstream licences.

⚠️ **Disclaimer**: For study and architecture research. It does not provide, contain or distribute any copyrighted audio or lyrics; obtain audio through lawful channels yourself.
