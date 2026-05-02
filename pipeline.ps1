#!/usr/bin/env pwsh
# pipeline_phase2.ps1 — RF-Genesis generation only

param(
    [string]$ScenarioFile = "$PSScriptRoot\scenario.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

# ── Defaults ──────────────────────────────────────────────────────────────────
if (-not $env:RFGENESIS_DIR)    { $env:RFGENESIS_DIR    = "../rfgen" }
if (-not $env:RFGEN_ENV_PROMPT) { $env:RFGEN_ENV_PROMPT = "a living room" }
if (-not $env:CONDA_ENV_RFGEN)  { $env:CONDA_ENV_RFGEN  = "rfgen" }

Write-Host "=== RF-Genesis Phase 2 Pipeline ==="
Write-Host "  RFGENESIS_DIR    : $env:RFGENESIS_DIR"
Write-Host "  CONDA_ENV_RFGEN  : $env:CONDA_ENV_RFGEN"

# ── Parse scenarios ───────────────────────────────────────────────────────────
$scenarios = Get-Content $ScenarioFile -Raw | ConvertFrom-Json |
    Where-Object { $_.name -and $_.desc }

if ($scenarios.Count -eq 0) {
    Write-Error "No scenarios found in $ScenarioFile"
    exit 1
}

Write-Host "  Scenarios found  : $($scenarios.Count)"

# ── RF-Genesis generation ─────────────────────────────────────────────────────
$phase2Failed = [System.Collections.Generic.HashSet[string]]::new()
$phase2Done   = 0
$phase2Total  = $scenarios.Count

Push-Location $env:RFGENESIS_DIR

try {

    foreach ($s in $scenarios) {

        $name = $s.name
        $desc = $s.desc

        $env_prompt = if ($null -ne $s.PSObject.Properties['env']) {
            $s.env
        } else {
            $env:RFGEN_ENV_PROMPT
        }

        $objDiff = Join-Path $env:RFGENESIS_DIR "output\$name\obj_diff.npz"

        $phase2Done++

        Write-Host "`n[$phase2Done/$phase2Total] $name"

        if (-not (Test-Path $objDiff)) {
            Write-Warning "  Missing obj_diff.npz — skipping"
            [void]$phase2Failed.Add($name)
            continue
        }

        Write-Host "  Running RF-Genesis..."

        & conda run -n $env:CONDA_ENV_RFGEN python run.py `
            -o $desc `
            -e $env_prompt `
            -n $name

        if ($LASTEXITCODE -ne 0) {
            Write-Warning "  RF-Genesis FAILED (exit $LASTEXITCODE)"
            [void]$phase2Failed.Add($name)
        }
        else {
            Write-Host "  Done: $name"
        }
    }

}
finally {
    Pop-Location
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host "`n=== Summary ==="
Write-Host "  Total scenarios : $($scenarios.Count)"
Write-Host "  Failed          : $($phase2Failed.Count)"

if ($phase2Failed.Count -gt 0) {
    Write-Host "`nFailures:"
    $phase2Failed | ForEach-Object {
        Write-Host "  - $_"
    }
}

if ($phase2Failed.Count -eq 0) {
    Write-Host "`nAll scenarios completed successfully."
    exit 0
}
else {
    Write-Host "`nCompleted with errors."
    exit 1
}