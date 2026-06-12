<#
.SYNOPSIS
    Reproduces the experiment on Windows using PowerShell.

.DESCRIPTION
    Creates and activates a Python virtual environment, installs exact
    dependencies from requirements_exact.txt, creates output directories,
    trains PPO and MaskablePPO models, evaluates checkpoints, summarizes
    results, and generates plots.

.PARAMETER PythonBin
    Python executable to use for creating the virtual environment.
    Defaults to "python". You can also pass "py -3.10" as a string
    when using the Python launcher, for example:
        .\reproduce.ps1 -PythonBin "py -3.10"
#>

param(
    [string]$PythonBin = $(if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }),
    [string]$VenvDir = ".venv"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-PythonBin {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    $parts = $PythonBin -split ' '
    $exe = $parts[0]
    $baseArgs = @()
    if ($parts.Count -gt 1) {
        $baseArgs = $parts[1..($parts.Count - 1)]
    }

    & $exe @baseArgs @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $PythonBin $($Arguments -join ' ')"
    }
}

Write-Host "Checking Python version..."
Invoke-PythonBin --version

Write-Host "Creating virtual environment if needed..."
if (-not (Test-Path -Path $VenvDir -PathType Container)) {
    Invoke-PythonBin -m venv $VenvDir
}

$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path -Path $ActivateScript -PathType Leaf)) {
    throw "Virtual environment activation script was not found: $ActivateScript"
}

Write-Host "Activating virtual environment..."
. $ActivateScript

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

if (-not (Test-Path -Path "requirements_exact.txt" -PathType Leaf)) {
    throw "requirements_exact.txt was not found in the current directory. Run this script from the project root."
}

Write-Host "Installing exact dependencies..."
pip install -r requirements_exact.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

Write-Host "Creating output directories..."
New-Item -ItemType Directory -Force -Path "results", "checkpoints", "logs" | Out-Null

Write-Host "Training models..."
python -m scripts.train --config "configs/train_ppo.yaml"
if ($LASTEXITCODE -ne 0) { throw "Training PPO failed." }
python -m scripts.train --config "configs/train_maskable_ppo.yaml"
if ($LASTEXITCODE -ne 0) { throw "Training MaskablePPO failed." }

Write-Host "Evaluating checkpoints..."
python -m scripts.eval --config "configs/eval.yaml"
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed." }

Write-Host "Summarizing results..."
python -m scripts.summarize --config "configs/summarize.yaml"
if ($LASTEXITCODE -ne 0) { throw "Summarization failed." }

Write-Host "Generating plots..."
python -m scripts.plot --config "configs/plot.yaml"
if ($LASTEXITCODE -ne 0) { throw "Plot generation failed for configs/plot.yaml." }
python -m scripts.plot --config "configs/plot_avg.yaml"
if ($LASTEXITCODE -ne 0) { throw "Plot generation failed for configs/plot_avg.yaml." }

Write-Host "Reproduction finished successfully."
