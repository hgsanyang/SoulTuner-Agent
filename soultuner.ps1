param(
    [ValidateSet("up", "down", "doctor", "test", "ingest", "logs", "mock", "netease-start", "netease-stop", "netease-status")]
    [string]$Action = "up",

    [ValidateSet("cpu", "gpu")]
    [string]$Profile
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# Profile resolution: an explicit -Profile wins; otherwise SOULTUNER_PROFILE from
# .env; otherwise cpu. Hardcoding gpu as the repo default would break every
# CPU-only clone, so a machine with a GPU opts in once in its own .env instead.
function Resolve-Profile {
    param([string]$Explicit)

    if ($Explicit) { return $Explicit }
    $fromEnv = (Get-ProjectEnvValue "SOULTUNER_PROFILE" "cpu").Trim().ToLowerInvariant()
    if ($fromEnv -notin @("cpu", "gpu")) {
        Write-Host "SOULTUNER_PROFILE='$fromEnv' 不是 cpu/gpu，按 cpu 处理"
        return "cpu"
    }
    return $fromEnv
}

function Invoke-ProjectPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    $env:PYTHONUTF8 = "1"
    $CondaPython = Join-Path $env:USERPROFILE "anaconda3\envs\music_agent\python.exe"
    if (Test-Path $CondaPython) {
        & $CondaPython @Arguments
        return
    }
    if ($env:CONDA_DEFAULT_ENV -eq "music_agent") {
        & python @Arguments
        return
    }
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        & conda run -n music_agent python @Arguments
        return
    }
    & python @Arguments
}

function Invoke-ProjectPytest {
    Invoke-ProjectPython -c "import pytest" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Invoke-ProjectPython -m pytest tests/unit/ -q
        return
    }
    Write-Host "pytest is not available in music_agent; falling back to system python."
    & python -m pytest tests/unit/ -q
}

function Assert-LastNativeCommand {
    param([string]$Step)

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed (exit code $LASTEXITCODE)."
    }
}

function Get-ProjectEnvValue {
    param([string]$Name, [string]$Default = "")

    $current = [Environment]::GetEnvironmentVariable($Name)
    if ($current) {
        return $current
    }
    $envPath = Join-Path $ProjectRoot ".env"
    if (-not (Test-Path $envPath)) {
        return $Default
    }
    $line = Get-Content -Path $envPath -ErrorAction SilentlyContinue |
        Where-Object { $_ -match "^\s*$([regex]::Escape($Name))\s*=" } |
        Select-Object -First 1
    if (-not $line) {
        return $Default
    }
    return (($line -split "=", 2)[1]).Trim().Trim('"').Trim("'")
}

function Get-NeteaseApiDir {
    # NETEASE_API_DIR env wins; otherwise look next to the repo, then in $HOME.
    # No hardcoded developer paths.
    $candidates = @(
        $env:NETEASE_API_DIR,
        (Join-Path $ProjectRoot "NeteaseCloudMusicApi"),
        (Join-Path (Split-Path $ProjectRoot -Parent) "tools\NeteaseCloudMusicApi"),
        (Join-Path $HOME "NeteaseCloudMusicApi")
    ) | Where-Object { $_ }
    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate "app.js")) {
            return $candidate
        }
    }
    return $null
}

function Get-NeteaseProcess {
    $conn = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue |
        Where-Object { $_.State -eq "Listen" } |
        Select-Object -First 1
    if (-not $conn) {
        return $null
    }
    return Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
}

function Show-NeteaseStatus {
    $proc = Get-NeteaseProcess
    if (-not $proc) {
        Write-Host "NeteaseAPI: stopped (:3000 is free)"
        return $false
    }
    Write-Host "NeteaseAPI: running on http://localhost:3000 (pid=$($proc.Id), process=$($proc.ProcessName))"
    return $true
}

function Start-NeteaseApi {
    if (Show-NeteaseStatus) {
        return
    }
    $dir = Get-NeteaseApiDir
    if (-not $dir) {
        throw "NeteaseCloudMusicApi not found. Set NETEASE_API_DIR, or place app.js under the project root, a sibling tools\NeteaseCloudMusicApi, or $HOME\NeteaseCloudMusicApi."
    }
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npm) {
        throw "npm.cmd not found. Install Node.js or start NeteaseCloudMusicApi manually."
    }
    Write-Host "Starting NeteaseAPI from $dir ..."
    Start-Process -FilePath $npm.Source -ArgumentList "start" -WorkingDirectory $dir -WindowStyle Hidden | Out-Null
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Seconds 1
        if (Show-NeteaseStatus) {
            return
        }
    }
    throw "NeteaseAPI did not start on :3000 within 20s."
}

function Stop-NeteaseApi {
    $containerId = docker compose --profile cpu --profile gpu --profile memory --profile rag ps -q netease 2>$null
    if ($LASTEXITCODE -eq 0 -and $containerId) {
        docker compose --profile cpu --profile gpu stop netease
        Assert-LastNativeCommand "Stopping Docker Netease proxy"
        Write-Host "Docker Netease proxy stopped"
        return
    }

    $proc = Get-NeteaseProcess
    if (-not $proc) {
        Write-Host "NeteaseAPI: already stopped"
        return
    }
    if ($proc.ProcessName -ne "node") {
        throw "Port 3000 belongs to $($proc.ProcessName) (pid=$($proc.Id)); refusing to stop an unrelated process."
    }
    Stop-Process -Id $proc.Id -Force
    Write-Host "NeteaseAPI stopped (pid=$($proc.Id))"
}

function Stop-LocalNeteaseApiForDocker {
    $containerId = docker compose --profile cpu --profile gpu --profile memory --profile rag ps -q netease 2>$null
    if ($LASTEXITCODE -eq 0 -and $containerId) {
        return
    }

    $proc = Get-NeteaseProcess
    if (-not $proc) {
        return
    }
    if ($proc.ProcessName -eq "node") {
        Write-Host "Stopping old local NeteaseAPI on :3000 before starting Docker proxy (pid=$($proc.Id))"
        Stop-Process -Id $proc.Id -Force
        Start-Sleep -Seconds 1
        return
    }
    Write-Warning "Port 3000 is occupied by $($proc.ProcessName) (pid=$($proc.Id)). Docker Netease proxy may not start."
}

function Assert-Neo4jEditionMatchesVolume {
    # Community image + Enterprise `block` volume = Neo4j refuses to start, and the
    # failure reads like a generic crash. Catch it before `up`, not after.
    # preflight_neo4j.py is pure stdlib so it runs under whatever Python we find.
    Invoke-ProjectPython "scripts/preflight_neo4j.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Neo4j preflight failed - see docs/NEO4J_MIGRATION.md. Nothing was started."
        exit 1
    }
}

$Profile = Resolve-Profile $Profile
Write-Host "Profile: $Profile"

switch ($Action) {
    "up" {
        Assert-Neo4jEditionMatchesVolume
        Stop-LocalNeteaseApiForDocker
        $ComposeFiles = @("-f", "docker-compose.yml")
        $MemoryBackends = (($env:MEMORY_EPISODIC_BACKENDS -split ",") | ForEach-Object { $_.Trim().ToLowerInvariant() })
        $UseGraphZep = $MemoryBackends -contains "graphzep"
        $KnowledgeVectorBackend = (Get-ProjectEnvValue "MUSIC_KNOWLEDGE_VECTOR_BACKEND" "qdrant").ToLowerInvariant()
        $EnableQdrantFlag = (Get-ProjectEnvValue "ENABLE_QDRANT" "").ToLowerInvariant()
        $DisableQdrantFlag = (Get-ProjectEnvValue "DISABLE_QDRANT" "").ToLowerInvariant()
        $UseQdrant = ($DisableQdrantFlag -ne "1") -and (
            $KnowledgeVectorBackend -eq "qdrant" -or $EnableQdrantFlag -eq "1" -or $KnowledgeVectorBackend -eq ""
        )
        if ($Profile -eq "gpu") {
            $ComposeFiles += @("-f", "docker-compose.gpu.yml")
            $env:DENSE_TEXT_AUDIO_BACKEND = "muq"
        } else {
            $env:DENSE_TEXT_AUDIO_BACKEND = "m2d"
        }
        if ($UseQdrant) {
            $env:MUSIC_KNOWLEDGE_VECTOR_BACKEND = "qdrant"
            docker compose @ComposeFiles --profile rag up -d qdrant
            Assert-LastNativeCommand "Starting optional Qdrant knowledge sidecar"
        }
        if ($Profile -eq "gpu") {
            # --build is required, not tidiness: `up` happily reuses an existing
            # CPU image and the overlay's cu124 build arg is then never applied,
            # so `up gpu` silently produces a CPU container.
            Write-Host "Building backend/ingest-worker with the CUDA overlay..."
            docker compose @ComposeFiles build backend ingest-worker
            Assert-LastNativeCommand "Building CUDA images"
        }
        docker compose @ComposeFiles --profile $Profile up -d --remove-orphans neo4j searxng netease backend
        Assert-LastNativeCommand "Starting core Docker services"
        if ($UseGraphZep) {
            docker compose @ComposeFiles --profile memory up -d graphzep
            Assert-LastNativeCommand "Starting optional GraphZep memory sidecar"
        }
        docker compose @ComposeFiles --profile $Profile up -d frontend
        Assert-LastNativeCommand "Starting frontend"
        if ($Profile -eq "gpu") {
            docker compose @ComposeFiles --profile gpu up -d ingest-worker
            Assert-LastNativeCommand "Starting GPU ingestion worker"
            # Asked for GPU, so prove it arrived — on BOTH services. Checking
            # only the backend leaves the exact case that costs most: the
            # long-running worker quietly extracting every vector on CPU.
            docker compose @ComposeFiles exec -T backend python scripts/assert_cuda.py
            Assert-LastNativeCommand "backend CUDA self-check"
            docker compose @ComposeFiles exec -T ingest-worker python scripts/assert_cuda.py
            Assert-LastNativeCommand "ingest-worker CUDA self-check"
        }
        Write-Host "Frontend: http://localhost:3003"
        Write-Host "Backend:  http://localhost:8501"
        Write-Host "Neo4j:    http://localhost:7474"
        if ($UseGraphZep) {
            Write-Host "GraphZep: http://localhost:3100 (optional sidecar)"
        } else {
            Write-Host "Memory:   local structured ledger + Neo4j hot path"
        }
        if ($UseQdrant) {
            Write-Host "Qdrant:   http://localhost:6333 (knowledge vector sidecar)"
        } else {
            Write-Host "RAG:      SQLite FTS only (set MUSIC_KNOWLEDGE_VECTOR_BACKEND=qdrant or ENABLE_QDRANT=1 to enable Qdrant)"
        }
        Write-Host "SearxNG:  http://localhost:8888"
        Write-Host "Netease:  http://localhost:3000"
    }
    "down" {
        docker compose --profile cpu --profile gpu --profile memory --profile rag down
        Assert-LastNativeCommand "Stopping Docker services"
    }
    "doctor" {
        Invoke-ProjectPython scripts/doctor.py
    }
    "test" {
        Invoke-ProjectPytest
    }
    "ingest" {
        if ($Profile -eq "gpu") {
            # Without -f docker-compose.gpu.yml the gpu *profile* selects a
            # service built from the CPU base and given no device — the profile
            # and the overlay are different mechanisms and both are needed.
            $IngestFiles = @("-f", "docker-compose.yml", "-f", "docker-compose.gpu.yml")
            docker compose @IngestFiles build ingest-worker
            Assert-LastNativeCommand "Building CUDA ingestion image"
            docker compose @IngestFiles --profile gpu run --rm ingest-worker python scripts/assert_cuda.py
            Assert-LastNativeCommand "ingest-worker CUDA self-check"
            docker compose @IngestFiles --profile gpu run --rm ingest-worker python scripts/ingest_worker.py
            Assert-LastNativeCommand "Running GPU ingestion"
        } else {
            Invoke-ProjectPython scripts/ingest_worker.py
        }
    }
    "logs" {
        docker compose --profile cpu --profile gpu --profile memory --profile rag logs -f --tail 200
    }
    "mock" {
        Invoke-ProjectPython scripts/dev/start_backend.py --mock
    }
    "netease-start" {
        Start-NeteaseApi
    }
    "netease-stop" {
        Stop-NeteaseApi
    }
    "netease-status" {
        Show-NeteaseStatus | Out-Null
    }
}
