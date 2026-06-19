param(
    [ValidateSet("up", "down", "doctor", "test", "ingest", "logs", "mock", "netease-start", "netease-stop", "netease-status")]
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

function Invoke-ProjectPytest {
    Invoke-ProjectPython -c "import pytest" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Invoke-ProjectPython -m pytest tests/unit/ -q
        return
    }
    Write-Host "pytest is not available in music_agent; falling back to system python."
    & python -m pytest tests/unit/ -q
}

function Get-NeteaseApiDir {
    $candidates = @(
        (Join-Path $ProjectRoot "NeteaseCloudMusicApi"),
        "C:\Users\sanyang\sanyangworkspace\tools\NeteaseCloudMusicApi",
        (Join-Path $HOME "NeteaseCloudMusicApi")
    )
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
        throw "NeteaseCloudMusicApi not found. Expected app.js under project root, C:\Users\sanyang\sanyangworkspace\tools\NeteaseCloudMusicApi, or $HOME\NeteaseCloudMusicApi."
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
    $proc = Get-NeteaseProcess
    if (-not $proc) {
        Write-Host "NeteaseAPI: already stopped"
        return
    }
    Stop-Process -Id $proc.Id -Force
    Write-Host "NeteaseAPI stopped (pid=$($proc.Id))"
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
        Invoke-ProjectPytest
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
