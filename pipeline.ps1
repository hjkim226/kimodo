#!/usr/bin/env pwsh
# run_pipeline.ps1 — Kimodo → RF-Genesis batch pipeline
#
# Environment variables (set before running, or edit defaults below):
#   RFGENESIS_DIR     : RF-Genesis repo root          (default: C:\RF-Genesis)
#   KIMODO_DIR        : directory with adapter script  (default: script directory)
#   RFGEN_ENV_PROMPT  : environment prompt for RFGen   (default: "a living room")
#   KIMODO_MODEL      : Kimodo model name              (default: Kimodo-SMPLX-RP-v1)
#   KIMODO_DURATION   : motion duration in seconds     (default: 5.0)
#   CONDA_ENV_RFGEN   : conda env name for rfgen       (default: rfgen)
#
# Usage:
#   $env:RFGENESIS_DIR = "D:\RF-Genesis"
#   .\run_pipeline.ps1
#   .\run_pipeline.ps1 -ScenarioFile "C:\other\scenario.json"

param(
    [string]$ScenarioFile = "$PSScriptRoot\scenario.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

# ── Defaults ──────────────────────────────────────────────────────────────────
if (-not $env:RFGENESIS_DIR)    { $env:RFGENESIS_DIR    = "C:\RF-Genesis" }
if (-not $env:KIMODO_DIR)       { $env:KIMODO_DIR       = $PSScriptRoot }
if (-not $env:RFGEN_ENV_PROMPT) { $env:RFGEN_ENV_PROMPT = "a living room" }
if (-not $env:KIMODO_MODEL)     { $env:KIMODO_MODEL     = "Kimodo-SMPLX-RP-v1" }
if (-not $env:KIMODO_DURATION)  { $env:KIMODO_DURATION  = "5.0" }
if (-not $env:CONDA_ENV_RFGEN)  { $env:CONDA_ENV_RFGEN  = "rfgen" }

$adapterScript = Join-Path $env:KIMODO_DIR "kimodo_smplx_to_rfgen_smpl.py"
$outputBase    = Join-Path $env:RFGENESIS_DIR "output"

Write-Host "=== Pipeline Configuration ==="
Write-Host "  RFGENESIS_DIR    : $env:RFGENESIS_DIR"
Write-Host "  KIMODO_DIR       : $env:KIMODO_DIR"
Write-Host "  RFGEN_ENV_PROMPT : $env:RFGEN_ENV_PROMPT"
Write-Host "  KIMODO_MODEL     : $env:KIMODO_MODEL"
Write-Host "  CONDA_ENV_RFGEN  : $env:CONDA_ENV_RFGEN"

# ── Parse scenarios from JSON ─────────────────────────────────────────────────
$scenarios = Get-Content $ScenarioFile -Raw | ConvertFrom-Json |
    Where-Object { $_.name -and $_.desc }   # skip empty placeholder entries

if ($scenarios.Count -eq 0) {
    Write-Error "No scenarios found in $ScenarioFile"
    exit 1
}
Write-Host "  Scenarios found  : $($scenarios.Count)`n"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if (-not (Test-Path $env:RFGENESIS_DIR)) {
    Write-Error "RFGENESIS_DIR not found: $env:RFGENESIS_DIR"
    exit 1
}
if (-not (Test-Path $adapterScript)) {
    Write-Error "Adapter script not found: $adapterScript"
    exit 1
}

# ── Start text encoder in background ─────────────────────────────────────────
Write-Host "Starting Kimodo text encoder in background..."
$encoderProc = Start-Process -FilePath "kimodo_textencoder" -PassThru -NoNewWindow
Start-Sleep -Seconds 2   # brief wait for encoder to initialize

# ── Phase 1: Motion generation + conversion (all scenarios first) ─────────────
Write-Host "`n=== Phase 1: Motion Generation & Conversion ($($scenarios.Count) scenarios) ==="

$phase1Failed = [System.Collections.Generic.HashSet[string]]::new()
$phase1Total  = $scenarios.Count
$phase1Done   = 0

foreach ($s in $scenarios) {
    $name       = $s.name
    $desc       = $s.desc
    $duration   = if ($s.duration) { $s.duration } else { $env:KIMODO_DURATION }
    $env_prompt = if ($s.env)      { $s.env }      else { $env:RFGEN_ENV_PROMPT }
    $amassOut   = Join-Path $PSScriptRoot "${name}_amass.npz"
    $rfgenOut   = Join-Path $outputBase "$name\obj_diff.npz"
    $phase1Done++

    Write-Host "`n[$phase1Done/$phase1Total] $name"
    Write-Host "  (1) Generating motion: $desc"

    & kimodo_gen $desc --model $env:KIMODO_MODEL --duration $duration --output $name
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "  kimodo_gen FAILED (exit $LASTEXITCODE) — skipping $name"
        [void]$phase1Failed.Add($name)
        continue
    }

    # Rename kimodo_out_amass.npz → {name}_amass.npz to preserve before next run overwrites it
    $defaultAmass = Join-Path $PSScriptRoot "kimodo_out_amass.npz"
    if (Test-Path $defaultAmass) {
        Move-Item $defaultAmass $amassOut -Force
        Write-Host "  Preserved: $amassOut"
    } else {
        Write-Warning "  kimodo_out_amass.npz not found — skipping $name"
        [void]$phase1Failed.Add($name)
        continue
    }

    Write-Host "  (2) Converting to RF-Genesis format..."
    $outDir = Split-Path $rfgenOut
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Set-Content -Path (Join-Path $outDir "prompt.txt") -Value "name: $name`ndesc: $desc`nenv: $env_prompt"
    & python $adapterScript $amassOut $rfgenOut --face-sensor
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "  Adapter FAILED (exit $LASTEXITCODE)"
        [void]$phase1Failed.Add($name)
    } else {
        Write-Host "  Saved: $rfgenOut"
    }
}

Write-Host "`nPhase 1 complete. Failed: $($phase1Failed.Count)/$phase1Total"

# ── Phase 2: RF-Genesis (all scenarios, after Phase 1 fully done) ─────────────
Write-Host "`n=== Phase 2: RF-Genesis Generation ($($scenarios.Count) scenarios) ==="

$phase2Failed = [System.Collections.Generic.HashSet[string]]::new()
$phase2Total  = $scenarios.Count
$phase2Done   = 0

Push-Location $env:RFGENESIS_DIR
try {
    foreach ($s in $scenarios) {
        $name       = $s.name
        $desc       = $s.desc
        $env_prompt = if ($s.env) { $s.env } else { $env:RFGEN_ENV_PROMPT }
        $phase2Done++

        if ($phase1Failed.Contains($name)) {
            Write-Host "`n[$phase2Done/$phase2Total] $name — SKIPPED (Phase 1 failed)"
            continue
        }

        Write-Host "`n[$phase2Done/$phase2Total] $name"
        Write-Host "  (3) Running RF-Genesis..."

        & conda run -n $env:CONDA_ENV_RFGEN python run.py `
            -o $desc `
            -e $env_prompt `
            -n $name

        if ($LASTEXITCODE -ne 0) {
            Write-Warning "  RF-Genesis FAILED (exit $LASTEXITCODE)"
            [void]$phase2Failed.Add($name)
        } else {
            Write-Host "  Done: $name"
        }
    }
} finally {
    Pop-Location
}

# ── Stop text encoder ─────────────────────────────────────────────────────────
if ($null -ne $encoderProc -and -not $encoderProc.HasExited) {
    Write-Host "`nStopping Kimodo text encoder (PID $($encoderProc.Id))..."
    $encoderProc.Kill()
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host "`n=== Pipeline Summary ==="
Write-Host "  Total scenarios : $($scenarios.Count)"
Write-Host "  Phase 1 failed  : $($phase1Failed.Count)"
Write-Host "  Phase 2 failed  : $($phase2Failed.Count)"

if ($phase1Failed.Count -gt 0) {
    Write-Host "  Phase 1 failures:"
    $phase1Failed | ForEach-Object { Write-Host "    - $_" }
}
if ($phase2Failed.Count -gt 0) {
    Write-Host "  Phase 2 failures:"
    $phase2Failed | ForEach-Object { Write-Host "    - $_" }
}

$totalFailed = ($phase1Failed.Count + $phase2Failed.Count)
if ($totalFailed -eq 0) {
    Write-Host "`nAll scenarios completed successfully."
    exit 0
} else {
    Write-Host "`nCompleted with $totalFailed error(s)."
    exit 1
}
