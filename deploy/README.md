# Deployment Files

The primary Docker entrypoints stay at the repository root:

- `docker-compose.yml`
- `docker-compose.gpu.yml`
- `Dockerfile`
- `soultuner.ps1`

This keeps the default user path short:

```powershell
.\soultuner.ps1 up cpu
.\soultuner.ps1 up gpu
```

`deploy/legacy/` only contains older single-service compose files used for local
debugging. New development should prefer the root compose stack.
