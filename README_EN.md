# 🎵 SoulTuner Agent

<p align="center">
  <img src="assets/logo.png" alt="logo" width="200" />
</p>

<p align="center">
  <strong>Multimodal Music Recommendation Agent — Hybrid RAG × Knowledge Graph × Long-term Memory</strong>
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
  <img src="https://img.shields.io/badge/tests-244_passed-brightgreen?logo=pytest" alt="Tests" />
  <img src="https://img.shields.io/badge/code_style-ruff-261230?logo=ruff" alt="Ruff" />
</p>

<p align="center">
  <a href="README.md">中文</a> | <a href="README_EN.md">English</a>
</p>

## 🎯 Discover Music with Natural Language, Let AI Truly Understand You

SoulTuner is a **locally-deployed** AI music recommendation agent. It's not just a simple "search → play" tool, but a personal DJ that **continuously learns your musical taste**:

- 🗣️ **Describe what you want to hear in natural language** — "I'm feeling really down today, I just want some quiet time alone." The system automatically identifies your emotion and scenario to recommend music that fits your current state.
- 🧠 **Understands you better the more you use it** — Every like, save, skip, and conversation silently builds your personalized music profile, making the next recommendation more accurate over time.
- 🌐 **Local library not enough? Real-time web search fallback** — The Catalog Gap Detector decides when local inventory or metadata is insufficient and supplements candidates online.
- 🗺️ **Immersive Music Journey** — Describe a story or scenario, and the AI will orchestrate a complete music journey with emotional arcs.
- ♻️ **Discover → Stage → Ingest** — Found a good song? It downloads to a "Pending" staging area first. Preview, then confirm ingestion with automatic acoustic analysis.

> 📖 For full features and interaction details, please refer to [Feature_Walkthrough.md](Feature_Walkthrough.md)
>
> Orchestrated via a LangGraph multi-node Agent workflow, integrating Neo4j, a MuQ-MuLan text-to-music anchor, M2D-CLAP / OMAR-RQ auxiliary representations, LLMs, and a MemoryGateway layer for multi-path retrieval, weighted RRF fusion, streaming recommendations, web fallback, music journeys, and a behavior data flywheel.

---

## 🚀 For Regular Users · Docker in 3 Steps

```powershell
Copy-Item .env.example .env
# Edit .env: at least fill NEO4J_PASSWORD and DASHSCOPE_API_KEY
.\soultuner.ps1 up cpu        # Full CPU experience: Neo4j + Backend + Frontend + optional memory sidecar + SearxNG + Netease proxy
.\soultuner.ps1 doctor        # Open http://localhost:3003 after health checks pass
```

Both CPU and GPU modes start the complete product workflow. The difference is multimodal quality and ingestion throughput: `up cpu` defaults to the lighter M2D text-to-music fallback while Neo4j, recommendation, web fallback, and the frontend remain fully available; with an NVIDIA GPU, use `.\soultuner.ps1 up gpu` to enable MuQ-MuLan fp16 as the primary text-to-music anchor and start the separate ingestion worker. Memory now defaults to the Neo4j hot path and no longer starts GraphZep by default. To experiment with natural-language episodic memory sidecars, set `MEMORY_EPISODIC_BACKENDS=graphzep` or `graphzep,mem0` in `.env` before startup.

Do not expose a personal music library directly as a public demo. Set `PUBLIC_DEMO_MODE=true` and configure `ADMIN_API_KEY` in `.env`; download, ingest, and delete operations are disabled or require `X-API-Key`. Public demos should use CC/open catalogs or mock audio.

### CC-only Gradio Public Demo

For public showcases, use the lightweight Gradio demo instead of exposing a personal library. It reads the MTG-Jamendo CC sample outside the repository by default, recommends by tags/scenarios, and never connects to the private Neo4j catalog or performs download/ingest/delete actions.

```powershell
python -m pip install -r requirements-demo.txt
$env:PUBLIC_DEMO_MODE = "1"
$env:PUBLIC_DEMO_DATA_DIR = "C:\Users\sanyang\sanyangworkspace\music_recommendation\data\mtg_sample"
python demos/public_gradio_app.py
```

Use this for Hugging Face Spaces / ModelScope showcases. The full product experience still runs through Docker + Next.js / FastAPI.

<details>
<summary>Local development / GPU ingestion / manual steps</summary>

- Local development: create the `music_agent` Conda environment, install Python dependencies and `web/` dependencies, then run `python startup_all.py`.
- Model cache: Docker allows the first run to download a missing HuggingFace text encoder so vector retrieval works out of the box. For fully offline runs, execute `python scripts/download_models.py` first and set `HF_OFFLINE=true` in `.env`.
- GPU ingestion: online recommendation does not require a GPU; lyrics tagging, audio vectors, and batch ingestion are handled by `.\soultuner.ps1 up gpu` or `.\soultuner.ps1 ingest gpu`.
- Manual ports: Neo4j `:7687`, Backend `:8501`, Frontend `:3003`; GraphZep `:3100` is only an optional MemoryGateway sidecar.

</details>

---

## ✨ Core Features

| Feature | Description |
|---|---|
| 🔀 **Hybrid RAG** | Three content recall paths (graph / dense / BM25), weighted RRF fusion; personalization and long-tail exploration are post-recall score adjustments |
| 🎵 **Multimodal Text-to-Music** | MuQ-MuLan is the Chinese-strong primary recall model, M2D-CLAP is the fallback, and OMAR-RQ adds acoustic similarity |
| 🧠 **Long-term Memory** | MemoryGateway: Neo4j behavior hot path + optional GraphZep/Mem0 sidecars, with editable and clearable learned preferences |
| 📊 **Coarse Rank + Explore** | Graph Affinity coarse ranking cutoff + Thompson Sampling cold-start exploration slots |
| 🤖 **Smart Intent Recognition** | Layered intent plan: `hard_constraints / soft_intent / hints` + multi-turn inheritance |
| 👤 **User Profile** | Frontend visual profile panel (Genre/Emotion/Scenario/Language) → Neo4j hot path + long-term memory sidecars |
| 🌐 **Web Search Fallback** | Enabled by default; mixes a few online candidates when local results are healthy, and triggers SearxNG/Tavily/Zhipu discovery + Netease playable resolution when the catalog falls short |
| 🎼 **Music Journey** | LLM Story → Emotion breakdown → Step-by-step retrieval, real-time SSE streaming |
| ♻️ **Data Flywheel** | Download → Stage → Preview → Confirm Ingest → Ingest queue → Tag extraction → Vector encoding → Neo4j |
| 📋 **Library Mgmt** | Pending staging area + queue status/retry + My Library full-graph management (search/play/tag edit/delete) |
| 📡 **SSE Streaming** | Real-time frontend rendering: thinking process → song cards → recommendation reasons |
| 🐳 **Docker Deployment** | `docker compose up` one-click full-stack startup |

---

## 🖼️ Feature Preview

<div align="center">
<h3>🎬 Explore SoulTuner's Features</h3>
<p>
  <a href="https://www.bilibili.com/video/BV11dQLBDEeF/">
    <img src="https://img.shields.io/badge/▶_Demo_—_Bilibili_|_bilibili.com/video/BV11dQLBDEeF-00A1D6?style=for-the-badge&logo=bilibili&logoColor=white&labelColor=FB7299" alt="Demo Video" />
  </a>
</p>
</div>

### 🏠 Home · 💬 Chat · 🎵 Recommend · 🎧 Player · 🗺️ Journey

<table>
  <tr>
    <td><img src="assets/首页.png" alt="Home" /></td>
    <td><img src="assets/对话页面.png" alt="Chat" /></td>
  </tr>
  <tr>
    <td><img src="assets/音乐推荐.png" alt="Recommendation" /></td>
    <td><img src="assets/播放页1.png" alt="Player" /></td>
  </tr>
  <tr>
    <td colspan="2"><img src="assets/音乐旅程.png" alt="Journey" /></td>
  </tr>
</table>

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│  Frontend (Next.js :3003)                                           │
│  React UI  ·  Global Audio Player  ·  Music Journey  ·  Settings   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ SSE
┌──────────────────────────────▼──────────────────────────────────────┐
│  Backend (FastAPI :8501)                                            │
│  SSE Streaming API  ·  Settings API  ·  Static Audio Server        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  LangGraph Agent (StateGraph)                                       │
│                                                                     │
│  start → MemoryGateway Recall → Planner (LLM) → Intent Router     │
│                                                                     │
│     ┌─────────┬─────────┬─────────┬──────────┐                     │
│     ▼         ▼         ▼         ▼          ▼                     │
│  search_songs  chat  acquire  gen_reco  journey                    │
│     │                                                               │
│     ▼                                                               │
│  Hybrid Retrieval ──→ LLM Explainer ──→ Pref Extract ──→ MemoryGateway Write → end │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Hybrid Retrieval Engine                                            │
│                                                                     │
│  GraphRAG · Dense KNN · BM25 · Catalog Gap / Web Fallback          │
│         └──────────────────┬───────────────────┘                   │
│                            ▼                                        │
│              Weighted RRF Fusion (keeps per-source ranks)            │
│                            ▼                                        │
│              Coarse Rank (Graph Affinity cutoff)                     │
│                            ▼                                        │
│              Thompson Sampling (cold-start exploration slots)        │
│                            ▼                                        │
│              Content-Anchor Rerank (Semantic+Acoustic normalized)   │
│                            ▼                                        │
│              MMR Multi-dim Diversity (λ=0.7)                       │
└─────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Storage Layer                                                      │
│  Neo4j (Graph + Vectors + Memory Hot Path) · Optional Memory Sidecars │
└─────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 14 + React 18 |
| **Agent** | LangGraph StateGraph (layered intent planning + multi-recall routing) |
| **Backend** | FastAPI + SSE Streaming |
| **Graph Database** | Neo4j 5.x (Native Vector Index + Graph Relations + User Behavior direct-write) |
| **Audio Embeddings** | MuQ-MuLan (primary text-to-music, 512d) + M2D-CLAP (semantic fallback/rerank, 768d) + OMAR-RQ (acoustic auxiliary, 1024d) |
| **LLMs** | Default `dashscope / qwen3.7-plus`; other providers are advanced overrides |
| **Long-term Memory**| MemoryGateway (Neo4j hot path + optional GraphZep/Mem0 episodic sidecars) |
| **Web Search** | SearxNG federated search + Tavily + Zhipu WebSearch |
| **Ranking Algorithm**| Content-anchor rerank (Semantic+Acoustic) + bounded post-recall adjustments + Thompson Sampling + MMR |
| **Context Management**| GSSC Token budget pipeline (Gather/Select/Structure/Compress + async pre-compression) |
| **Containerization** | Docker Compose CPU/GPU entrypoints; CPU includes the full online stack, GPU adds the ingestion worker |

> 📖 See [tests/eval/README.md](tests/eval/README.md) for recommendation-quality and alignment evaluation commands.

---

## 🔬 Technical Depth

### RAG Hybrid Retrieval Pipeline

```text
User Query → Planner (LLM) outputs a layered plan
               ↓  hard_constraints + soft_intent + hints + intent_type
    ┌──────────┬──────────┬──────────┐
    ▼          ▼          ▼
 GraphRAG   Dense KNN    BM25               ← Step 1: content recall
 (Neo4j)   (MuQ+OMAR) (title/artist/lyrics)
    └──────────┴──────────┴──────────┘
               ▼
   Step 2: Weighted RRF fusion            ← Preserves per-source rank and source metadata
               ▼
   Step 3: hard_constraints + DISLIKES    ← Only hard filter; mood/scenario/genre stay soft
               ▼
   Step 4: Artist Diversity Filter        ← ≤ N songs per artist (exception for specific queries)
               ▼
   Step 5: Post-recall Adjust + Explore  ← Personal/fresh/long-tail boosts, time-decayed exposure penalty
               ▼
   Step 6: Content-Anchor Normalized Rerank ← Primary text-to-music anchor(MuQ/M2D fallback) + seed acoustic anchor(OMAR-RQ)
               ▼
   Step 7: MMR Multi-dim Diversity + FinalCut
```

**Key Design Decisions**:

- **Layered Intent Plan**: The Planner outputs `hard_constraints / soft_intent / hints`. Entities, language, and instrumental constraints are hard filters; mood, scenario, and vibe are ranking signals.
- **Three Content Recall Paths**: Graph entities/tags, MuQ-MuLan text-to-music search, and BM25 lexical search always run; `intent_type` plus query profile only adjust these content-source weights.
- **Weighted RRF Fusion**: Candidates are merged with `weight / (60 + rank)`, preserving source rank and source metadata instead of equal merging.
- **Three-model roles**: MuQ-MuLan is the default text-to-music anchor and the preferred semantic rerank anchor; M2D-CLAP remains available as recall/rerank fallback; OMAR-RQ is used for seed-based acoustic similarity when the user asks for songs similar to a reference track.
- **Personalization and cold-start are post-recall score adjustments**: User profile, long-tail, freshness, and time-decayed exposure affect only content-recalled candidates. They are neither independent recall sources nor duplicated inside the content anchor.
- **Unified adjustment scale**: Personalization, freshness, long-tail, and exposure fatigue are normalized to `[0,1]` and merged into a clipped delta (default `±0.08`) so content relevance remains dominant.
- **Catalog Gap Detector**: Web search is enabled by default. Healthy local results get a small online mix-in (default `WEB_MIX_IN_COUNT=4`) to expand the library. Era/latest/external-knowledge requests or weak local inventory switch to fallback mode (default `WEB_FALLBACK_COUNT=10`). If web search is disabled, the agent does not go online; it explains the local catalog boundary and asks the user to enable web search for broader results.
- **Two-stage online candidate resolution**: For knowledge-heavy gaps, SearxNG/Tavily/Zhipu first discover context-relevant title/artist candidates, then the Netease proxy resolves playable songs. Liking or saving an online song downloads it to the Pending staging area for user-confirmed ingestion.
- **Coarse Rank + Thompson Sampling**: Content RRF plus the clipped adjustment delta is used for cutoff (`coarse_cut_ratio=65%`), while tail candidates are rescued via TS sampling (`Beta(α,β)` distribution) for long-tail/new-song exploration.
- **Content-Anchor Normalized Reranking**: Semantic reranking follows the configured primary dense backend (MuQ by default, M2D fallback). The acoustic anchor no longer uses the candidate-set centroid; OMAR-RQ only aligns to explicit seed tracks. Personalization is applied only through the bounded post-recall layer instead of a 25% ranking anchor.
- **Deployable fallback**: `DENSE_TEXT_AUDIO_BACKEND=muq|m2d|both`; a missing MuQ model/index or encoding failure automatically falls back to M2D, while lazy loading keeps the default memory footprint bounded.
- **Explicit DST + PlanDelta**: First turns and topic resets use the full planner. Established follow-ups emit only whitelisted `add/replace/remove/clear_topic` operations, while deterministic code preserves untouched slots. Unresolved references, missing key entities, or severe conflicts return a clarification question with options; delta failures fall back to the full planner.
- **MMR Jaccard**: Re-ranking using the `{genre, mood, theme, scenario}` multidimensional tags for candidate diversity.

### Agent Workflow

```mermaid
stateDiagram-v2
    [*] --> recall_memory: Start
    recall_memory --> plan_query: MemoryGateway Profile/Memory Recall
    plan_query --> route_intent: LLM Structured Output (layered intent)

    route_intent --> search_songs: graph_search / hybrid_search / vector_search
    route_intent --> web_fallback: web_search (direct to live search)
    route_intent --> chat_response: general_chat
    route_intent --> acquire_music: acquire_music
    route_intent --> gen_recommendations: recommend_by_favorites
    route_intent --> clarify: clarification_required

    search_songs --> web_fallback: Catalog Gap fallback / online mix-in
    search_songs --> explain_results: Push songs first on local hit
    web_fallback --> explain_results: Online Music API live search
    acquire_music --> explain_results
    gen_recommendations --> explain_results

    explain_results --> extract_preferences: Async tuner-style response
    extract_preferences --> persist_memory: LLM Preference Extraction (Async)

    chat_response --> persist_memory: General chat skips preference extraction
    clarify --> persist_memory: Persist pending clarification state

    persist_memory --> [*]: MemoryGateway Async Write
```

> Intent recognition, HyDE, and the async tuner-style response default to `dashscope / qwen3.7-plus`. Recommended songs are streamed first; the default `EXPLANATION_MODE=tuner_async` then generates a short conversational response plus follow-up chips. The legacy per-song explanation mode remains available via `EXPLANATION_MODE=song_detail`, but is not recommended as the default because the system should not invent detailed listening impressions.

> `web_search` intent now routes **directly to the `web_fallback` node** (Online Music API live search), bypassing HybridRetrieval entirely. Normal recommendation can still use the Catalog Gap Detector to mix in a few online songs, while era/latest/external-knowledge gaps or weak local inventory trigger fallback. Supports Chinese-first query extraction, multi-level fallback query resolution, 30-second preview detection, and a two-stage "web evidence discovery → Netease playable resolution" path.

> Preference extraction is governed by a decoupled `extract_preferences` node; general chat intents automatically bypass it.

### Memory System

| Component | Description |
|---|---|
| **MemoryGateway** | Unified memory entry point; Neo4j stores behavior/structured hot-path preferences, while GraphZep/Mem0 can dual-run as optional sidecars |
| **GSSC Token Budget** | Dynamic memory assignment for facts + chat_history, LLM summarization + async pre-compression caching |
| **Neo4j Preference Graph** | Auto-extract user preferences from chat, async write to Neo4j User nodes; behavior events (like/save/skip/dislike) directly write relationship edges |
| **Editable User Profile** | Frontend profile panel edits declared preferences and can remove one learned preference or clear all learned avoid/context/discovery preferences |
| **Profile Synthesizer** | Dynamic profile synthesis: aggregates long-term memory + behavioral stats (played/liked/skipped counts) → auto-generates a structured user portrait injected into each Planner prompt |

**Memory Architecture Highlights**:
- **Neo4j** handles precise behavioral relationships (LIKES / SAVES / LISTENED_TO / SKIPPED / DISLIKES) with fast Bolt direct-write (~100ms)
- **GraphZep / Mem0** only manage fuzzy semantic memory sidecars. Configure `MEMORY_EPISODIC_BACKENDS=graphzep,mem0` to dual-run them; disabling sidecars does not break the recommendation hot path.
- **Profile Synthesizer** asynchronously aggregates both memory sources per conversation turn, generating a readable `portrait` injected into the current Planner system prompt

### User Profile System

The frontend profile panel saves preferences (genre/mood/scenario/language) to the Neo4j `User` node and sends natural-language memory to MemoryGateway sidecars. It also exposes learned avoid/context/discovery preferences for single-item deletion or full learned-memory clearing without deleting manual profile settings, likes, or saves. Graph Affinity reads these attributes during retrieval to boost favored tracks or softly demote avoided ones. The Profile Synthesizer aggregates behavior statistics and memory snapshots to provide personalized context injection.

### Data Flywheel

User search → Discover new song → Download to "Pending" staging area → Frontend preview & playback → Select and confirm ingestion → Ingest queue with pending/processing/done/failed status and retry → LLM label extraction + MuQ/M2D/OMAR vector encoding → Neo4j ingestion → Discoverable next time.

> 💡 Songs acquired from the web no longer auto-ingest. Users manage ingested songs from the "My Library" page (search/play/tag edit/delete).

### Feedback Loop

Every recommendation slate is written under `${MUSIC_DATA_PATH}/feedback` with a query hash, intent, source ranks, content-anchor scores, post-recall components, and final position; raw queries are disabled by default. The frontend returns the exact `exposure_id` and position with like/save/skip/full-play/repeat/dislike events. Play-start is neutral, and untouched impressions are never mislabeled as negatives.

```powershell
python scripts/replay_feedback.py
python scripts/replay_feedback.py --write-candidate
python scripts/replay_feedback.py --promote
python scripts/replay_feedback.py --rollback
```

Replay uses `explicit_feedback_logistic_v2`: strict exposure attribution, chronological train/validation splitting, baseline-vs-candidate metrics, and same-exposure preference pairs. A candidate must pass validation before promotion; active and previous versions support rollback. Learned policies are bounded to `0.8~1.2` multipliers for RRF, content anchors, and post-recall terms while preserving the `±0.08` adjustment cap. Global and sufficiently supported per-user policies remain isolated. Insufficient real positive/negative feedback returns `insufficient_data` and never activates fake weights.

### Engineering Quality

| Dimension | Description |
|---|---|
| **CI/CD** | GitHub Actions — Auto runs `ruff` linting and `pytest` unit tests |
| **Unit Testing** | 244 tests covering settings loading, Planner/Delta Planner, outcome eval, fusion filters, DST, strict feedback attribution, policy rollback, alignment calibration, public demo, teacher logs, and more |
| **Outcome Eval** | `evaluate_outcomes` measures whether returned songs satisfy the user's intent; `context_dev/context_holdout` add a Chinese context-matching ruler with 11 goal categories and 4 specificity levels |
| **Token Tracking** | Built-in structured Token consumption reports in GSSC pipelines |
| **State Persistence** | LangGraph MemorySaver Checkpoint (in-memory, replaceable with DB adapters) |
| **Code Standards** | Enforced by Ruff static analysis + pyproject.toml |

<details>
<summary>Outcome Eval Details</summary>

```powershell
python -m tests.eval.evaluate_outcomes --split dev --planner-temperature 0 --fast
python -m tests.eval.evaluate_outcomes --split holdout --planner-temperature 0 --fast
python -m tests.eval.evaluate_outcomes --split holdout_hard --planner-temperature 0 --fast
python -m tests.eval.evaluate_outcomes --split holdout_easy --planner-temperature 0 --fast --require-no-failures
python -m tests.eval.evaluate_outcomes --split context_dev --planner-temperature 0 --fast --case-timeout 75
python -m tests.eval.evaluate_outcomes --split context_holdout --planner-temperature 0 --fast --case-timeout 75
python -m tests.eval.evaluate_outcomes --split dev --planner-temperature 0 --fast --timing --case-timeout 45
```

The harness checks whether returned songs satisfy artist, title, language, playability, negation, soft intent, and fallback behavior. The `context_*` splits are the new non-saturated Chinese ruler for "understand this moment and select fitting songs"; run them with `--planner-temperature 0`.

The legacy `evaluate_intent.py` remains useful only as a route-label regression check; it is no longer used as evidence of recommendation quality. See `tests/eval/README.md` for details.

</details>

### Distillation Teacher Logs

Planner, HyDE, and tuner-style explanation outputs can be written locally for later SFT/distillation. Logging is disabled by default; when enabled, text fields are hashed unless explicitly allowed:

```powershell
$env:TEACHER_LOG = "1"
$env:TEACHER_LOG_DIR = "data/teacher"
```

Only set `TEACHER_LOG_STORE_TEXT=1` when building a private local training set. Do not commit `data/teacher/` to the remote repository.

---

## 📊 Neo4j Knowledge Graph

```mermaid
erDiagram
    Song ||--|| Artist : PERFORMED_BY
    Song }o--o{ Genre : HAS_GENRE
    Song }o--o{ Mood : HAS_MOOD
    Song }o--o{ Scenario : FITS_SCENARIO
    Song ||--|| Language : IN_LANGUAGE
    Song ||--|| Region : IN_REGION
    User }o--o{ Song : "LIKES / SAVES / LISTENED_TO"
    User }o--o{ Genre : PREFERS_GENRE

    Song {
        string title
        string music_id
        float_arr muq_embedding
        float_arr m2d2_embedding
        float_arr omar_embedding
        string audio_url
    }
    Artist { string name }
    Genre { string name }
    Mood { string name }
    Scenario { string name }
    User {
        string user_id
        string preferred_genres
        string preferred_moods
        string preferred_scenarios
        string preferred_languages
        string profile_free_text
    }
```

**Vector Indices**: `song_muq_index` (512d, cosine, primary text-to-music) + `song_m2d2_index` (768d, cosine, fallback/rerank) + `song_omar_index` (1024d, cosine, acoustic auxiliary).

---

## 🧰 Startup And Ingestion Reference

For everyday use, the 3-step Docker block near the top is enough. The commands below keep only two startup modes: CPU and GPU.

| Command | Purpose |
|---|---|
| `.\soultuner.ps1 up cpu` | Full CPU online stack; text-to-music uses the M2D fallback by default to avoid heavy MuQ cold loading |
| `.\soultuner.ps1 up gpu` | Full CPU workflow + MuQ-MuLan fp16 primary text-to-music anchor + separate ingestion worker for lyrics tags, audio vectors, and batch ingest |
| `MEMORY_EPISODIC_BACKENDS=graphzep` | Optionally enables the GraphZep natural-language memory sidecar without changing the Neo4j hot path |
| `.\soultuner.ps1 doctor` | Environment diagnosis and next-step hints |
| `.\soultuner.ps1 test` | Unit tests |
| `.\soultuner.ps1 mock` | End-to-end mock run without external services |
| `.\soultuner.ps1 ingest gpu` | Process the ingestion queue with the GPU worker |

<details>
<summary>First-time model cache and data ingestion</summary>

```powershell
conda create -n music_agent python=3.11
conda activate music_agent
pip install -r requirements.txt
python scripts/download_models.py
.\soultuner.ps1 ingest gpu
```

Online recommendation only reads mounted model caches. Batch lyrics tagging, audio-vector extraction, and new-song ingestion are handled by the GPU worker so day-to-day recommendation does not depend on GPU work.

Manual yt-dlp downloads under `data/yt_dlp_manual/downloads` use the same canonical ingestion path:

```powershell
python data/pipeline/yt_dlp_manual_flywheel.py --dry-run
python data/pipeline/yt_dlp_manual_flywheel.py --stage --include-existing --batch-size 5
python data/pipeline/ingest_to_neo4j.py --dataset yt_dlp_manual --manifest data/pipeline/gemini_prompts/<manifest>.json --force
```

The flywheel stages files into `../data/processed_audio/`, generates LLM tags, then uses GPU extraction for MuQ/M2D/OMAR before writing to Docker Neo4j. Generated manifests, tag JSON, and media files are gitignored.

</details>

<details>
<summary>Local development / manual startup</summary>

```powershell
cd web
npm install
cd ..
python startup_all.py
```

| Service | Port |
|---|---|
| Neo4j Bolt / Browser | `7687` / `7474` |
| GraphZep | `3100` (optional memory profile) |
| Backend | `8501` |
| Frontend | `3003` |
| SearxNG | `8888` |

</details>

---
### Advanced: Local LLM Experiment (Optional)

DashScope is the recommended default. Switch providers only when you explicitly want to test a local model, from **Settings → Model Config → Advanced Options**.

1. **Terminal A (WSL)**: Start the local inference engine.

   ```bash
   wsl
   bash /path/to/SoulTuner-Agent/scripts/start_sglang.sh
   ```

2. **Frontend Advanced Options**: switch the specific model slot to `sglang`, then save.

---

## 📁 Project Structure

```
.
├── agent/                      # LangGraph Agent
│   ├── music_agent.py          # Native agent loop
│   └── music_graph.py          # StateGraph workflow with layered intent routing
│
├── api/                        # FastAPI Interfaces
│   ├── server.py               # Gateway & Settings API
│   └── user_profile.py         # User Preferences API (GET/POST /api/user-profile)
│
├── config/settings.py          # Global Pydantic configs (Runtime patchable)
│
├── retrieval/                  # Engine abstractions
│   ├── hybrid_retrieval.py     # Multi-path Fusion + bounded adjustment/TS + Content-Anchor Rerank + MMR
│   ├── gssc_context_builder.py # GSSC pipeline (Budgeting + Abstract Context mapping)
│   ├── muq_embedder.py         # MuQ-MuLan 24kHz audio/text encoder (lazy-loaded)
│   ├── audio_embedder.py       # M2D-CLAP fallback and semantic rerank encoder
│   ├── neo4j_client.py         # Node connectivity definitions
│   ├── music_journey.py        # Journey architect algorithms
│   └── user_memory.py          # Neo4j Preferences & Logs
│
├── tools/                      # Tool executions
│   ├── graphrag_search.py      # Neo4j Cypher definitions
│   ├── semantic_search.py      # MuQ primary, M2D fallback, OMAR-assisted retrieval
│   ├── web_search_aggregator.py# SearxNG + Tavily routers
│   └── acquire_music.py        # Flywheel tools (download to staging + on-demand ingest)
│
├── llms/                       # LLMs
│   ├── prompts.py              # LLM Prompts
│   ├── registry.py             # Provider registry + env injection
│   ├── chat_models.py          # LangChain ChatModel factories
│   ├── native.py               # Native LiteLLM caller
│   └── multi_llm.py            # Backward-compatible facade
│
├── schemas/                    # Pydantic schemas
├── services/                   # Outer microservice bindings
├── data/pipeline/              # DB ingest pipelines
├── web/                        # Next.js Frontend
│   ├── components/Settings/    # ⚙️ Settings interface
│   ├── components/Profile/     # 👤 User Profile interface
│   └── components/Navigation/  # Nav layout views
│   └── app/library/            # Library pages (Pending / My Library / Likes / Collections)
│
├── graphzep_service/           # Optional GraphZep memory sidecar
├── tests/                      # Testing & Eval
│   ├── unit/                   # 244 pytest tests
│   └── eval/                   # Outcome eval harness (evaluate_outcomes.py)
├── .github/workflows/ci.yml    # GitHub Actions definitions
├── docker-compose.yml          # Container configuration
├── Dockerfile                  # API Engine definitions
├── pyproject.toml              # Ruff + Pytest syntax bounds
├── .env.example                # Templates
└── startup_all.py              # OS unified boot pipeline
```

---

## ⚙️ Configuration

### Environment Variables

| Property | Description | Default |
|---|---|---|
| `DASHSCOPE_BASE_URL` | DashScope API base | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `DASHSCOPE_API_KEY` | DashScope API key | — |
| `MODEL_NAME` | Main reasoning model | `qwen3.7-plus` |
| `NEO4J_URI` | Neo4j bindings | `neo4j://127.0.0.1:7687` |
| `NEO4J_PASSWORD` | Neo4j security parameters | — |
| `DENSE_TEXT_AUDIO_BACKEND` | Text-to-music backend | `muq` (Docker CPU automatically uses `m2d`; `both` optional) |
| `MUSIC_DENSE_QUERY_VARIANTS` | Multi-view HyDE vector ensemble | `auto` (scene/mood/similarity queries use acoustic/emotional/context views; exact artist/title queries stay single-vector) |
| `MUSIC_ALIGNMENT_CALIBRATION_PATH` | Optional text/audio gap calibration JSON | empty (scale/bias per backend; missing file is no-op) |
| `MUSIC_FEEDBACK_DIR` | Exposure/event feedback log directory | `data/feedback` |
| `FEEDBACK_LOG_RAW_QUERY` | Store raw queries in feedback logs (hash-only by default) | `0` |
| `RECALL_SOURCE_TIMEOUT_SECONDS` | Per-recall timeout | `60` (covers MuQ cold loading) |
| `TAVILY_API_KEY` | Cloud indexing rules | Optional |

MuQ-MuLan processes 24kHz audio and loads on demand. This project measured about 2.75GB peak VRAM in fp32 and about 1.4GB in fp16. `up gpu` exposes the GPU to the backend and enables fp16, while `up cpu` selects M2D to avoid an unresponsive CPU container during MuQ cold loading. MuQ weights use **CC-BY-NC 4.0** and therefore are restricted to non-commercial use unless separately licensed.

---

## 🙏 Acknowledgements

Architectural inspiration was expanded heavily upon [imagist13/Muisc-Research](https://github.com/imagist13/Muisc-Research).

| Project | Purpose |
|---|---|
| [aexy-io/graphzep](https://github.com/aexy-io/graphzep) | Core graph storage structure representations |
| [OpenMuQ/MuQ](https://github.com/OpenMuQ/MuQ) | MuQ-MuLan primary text-to-music model (CC-BY-NC 4.0) |
| [nttcslab/m2d](https://github.com/nttcslab/m2d) | M2D-CLAP fallback and auxiliary semantics |
| [MTG/omar](https://github.com/MTG/omar) | Raw acoustics implementations |

---

## 📚 References

1. Niizumi, D. et al. (2025). *M2D-CLAP: Exploring General-purpose Audio-Language Representations Beyond CLAP.*
2. Alonso-Jiménez, P. et al. (2025). *OMAR-RQ: Open Music Audio Representation Model Trained with Multi-Feature Masked Token Prediction.*
3. Rasmussen, P. et al. (2025). *Zep: A Temporal Knowledge Graph Architecture for Agent Memory.*
4. Palumbo, E. et al. (Spotify, 2025). *You Say Search, I Say Recs: A Scalable Agentic Approach to Query Understanding and Exploratory Search.* (RecSys 2025)
5. D'Amico, E. et al. (Spotify, 2025). *Deploying Semantic ID-based Generative Retrieval for Large-Scale Podcast Discovery at Spotify.*
6. Penha, G. et al. (2025). *Semantic IDs for Joint Generative Search and Recommendation.* (RecSys 2025 LBR)
7. Palumbo, E. et al. (2025). *Text2Tracks: Prompt-based Music Recommendation via Generative Retrieval.*
8. Xu, S. et al. (2025). *Climber: Toward Efficient Scaling Laws for Large Recommendation Models.*
9. Wang, S. et al. (2025). *Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation.* (ACL 2025)

---

## 📄 License

MIT License

⚠️ **Disclaimer**: Produced and maintained solely for machine learning research applications and architectural experimentation limits. **Strictly NO commercial use**. Does not offer indexing mechanisms for commercialized data files.
