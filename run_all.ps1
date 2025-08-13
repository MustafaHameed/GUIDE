<#
.SYNOPSIS
  End-to-end runner for the 10-Aug-25 repo.

.DESCRIPTION
  - Installs requirements (if requirements.txt exists)
  - Runs EDA → Train → Nested CV → Tests → (optionally) Dashboard
  - Exits on the first error
  - Supports classification or regression workflows
  - Ensures the correct Python interpreter and env vars

.EXAMPLES
  # Full classification workflow
  .\run_all.ps1

  # Full regression workflow
  .\run_all.ps1 -Task regression

  # Skip dashboard (non-interactive CI-ish run)
  .\run_all.ps1 -SkipDashboard

  # Use a different Python interpreter
  .\run_all.ps1 -Python "C:/ProgramData/anaconda3/python.exe"
#>

param(
  [ValidateSet('classification', 'regression')]
  [string]$Task = 'classification',

  [string]$Python = "C:/ProgramData/anaconda3/python.exe",

  [switch]$SkipEDA,
  [switch]$SkipTrain,
  [switch]$SkipNestedCV,
  [switch]$SkipTests,
  [switch]$SkipDashboard
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# ----- Helpers -----
function Fail([string]$Message) {
  Write-Host "❌ $Message" -ForegroundColor Red
  exit 1
}

function Run-Step([string]$Name, [string]$Cmd, [switch]$Skip) {
  if ($Skip) {
    Write-Host "⏭  Skipping: $Name"
    return
  }
  Write-Host "▶  $Name"
  Write-Host "    $Cmd" -ForegroundColor DarkGray
  try {
    Invoke-Expression $Cmd
    Write-Host "✅ $Name completed"
  } catch {
    Fail "$Name failed: $($_.Exception.Message)"
  }
}

# ----- Move to repo root (where this script lives) -----
try { Set-Location -Path $PSScriptRoot } catch { Fail "Cannot cd to script directory." }

# ----- Sanity checks -----
if (-not (Test-Path $Python)) { Fail "Python not found at: $Python" }

# ----- Env vars (avoid MKL/KMeans issue; make src importable) -----
$env:OMP_NUM_THREADS = "2"
$env:PYTHONPATH = (Get-Location).Path

# ----- Requirements (optional) -----
if (Test-Path "requirements.txt") {
  Run-Step "Install requirements" "`"$Python`" -m pip install -r requirements.txt"
} else {
  Write-Host "ℹ  requirements.txt not found — continuing."
}

# ----- Sequence -----
Run-Step "EDA"           "`"$Python`" src/eda.py" -Skip:$SkipEDA

# Training (classification with fairness groups; tweak as needed)
if ($Task -eq 'classification') {
  Run-Step "Train (classification)" "`"$Python`" -m src.train --task classification --group-cols sex school" -Skip:$SkipTrain
} else {
  Run-Step "Train (regression)"     "`"$Python`" -m src.train --task regression" -Skip:$SkipTrain
}

Run-Step "Nested CV"    "`"$Python`" src/nested_cv.py" -Skip:$SkipNestedCV
Run-Step "Tests"        "`"$Python`" -m pytest -q"   -Skip:$SkipTests

# Dashboard (interactive; runs last)
if (-not $SkipDashboard) {
  Write-Host "🖥  Launching dashboard (Ctrl+C to stop)..."
  # Use -m to avoid PATH issues with streamlit
  & $Python -m streamlit run dashboard.py
  if ($LASTEXITCODE -ne 0) { Fail "Dashboard exited with code $LASTEXITCODE" }
} else {
  Write-Host "⏭  Skipping: Dashboard"
}

Write-Host "🎉 All requested steps completed successfully."
