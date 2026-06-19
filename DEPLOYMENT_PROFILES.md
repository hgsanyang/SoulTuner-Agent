# Deployment Profiles

SoulTuner uses one codebase with three operating profiles.

| Profile | Services | GPU |
| --- | --- | --- |
| Lite | Neo4j, API, frontend, cloud LLM | Not required |
| Standard | Lite plus GraphZep and SearxNG | Not required |
| Full | Standard plus offline ingestion worker | Recommended |

Windows commands:

```powershell
.\soultuner.ps1 doctor
.\soultuner.ps1 up lite
.\soultuner.ps1 up standard
.\soultuner.ps1 up full
.\soultuner.ps1 test
.\soultuner.ps1 ingest full
.\soultuner.ps1 mock
.\soultuner.ps1 netease-status
.\soultuner.ps1 netease-start
.\soultuner.ps1 netease-stop
.\soultuner.ps1 down
```

`mock` starts the backend with deterministic planner and retrieval fixtures. It
does not require an API key, Neo4j, GraphZep, or local embedding models.

Online song discovery remains available in every profile. Newly downloaded songs are
written to Neo4j immediately with playable URLs. Lyrics analysis and M2D-CLAP/OMAR
audio embeddings are queued under `data/ingest_queue/` and processed by the Full
profile worker, so the online API process does not need a GPU.

Stable local ports:

- Frontend: `3003`
- Backend API: `8501`
- GraphZep: `3100`
- Neo4j Browser/Bolt: `7474` / `7687`
- SearxNG: `8888`
- Compatible music API: `3000` when installed separately

`netease-start` looks for `NeteaseCloudMusicApi/app.js` under the project root,
`C:\Users\sanyang\sanyangworkspace\tools\NeteaseCloudMusicApi`, then
`$HOME\NeteaseCloudMusicApi`.
