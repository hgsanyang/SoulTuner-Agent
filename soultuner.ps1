param(
    [ValidateSet("up", "down", "doctor", "test", "ingest", "logs", "mock")]
    [string]$Action = "up",

    [ValidateSet("lite", "standard", "full")]
    [string]$Profile = "standard"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

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

switch ($Action) {
    "up" {
        if ($Profile -eq "lite") {
            docker compose up -d neo4j backend frontend
        } else {
            docker compose --profile $Profile up -d
        }
        Write-Host "Frontend: http://localhost:3003"
        Write-Host "Backend:  http://localhost:8501"
        Write-Host "Neo4j:    http://localhost:7474"
        if ($Profile -ne "lite") {
            Write-Host "GraphZep: http://localhost:3100"
            Write-Host "SearxNG:  http://localhost:8888"
        }
    }
    "down" {
        docker compose --profile standard --profile full down
    }
    "doctor" {
        Invoke-ProjectPython scripts/doctor.py
    }
    "test" {
        Invoke-ProjectPython -m pytest tests/unit/ -q
    }
    "ingest" {
        if ($Profile -eq "full") {
            docker compose --profile full run --rm ingest-worker python scripts/ingest_worker.py
        } else {
            Invoke-ProjectPython scripts/ingest_worker.py
        }
    }
    "logs" {
        docker compose --profile standard --profile full logs -f --tail 200
    }
    "mock" {
        Invoke-ProjectPython start.py --mock
    }
}
