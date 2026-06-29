# Run Jetson profiling remotely from Windows (requires OpenSSH client).
#
# Usage:
#   $env:JETSON_HOST = "192.168.1.50"
#   $env:JETSON_USER = "jetson"
#   .\scripts\run_jetson_remote.ps1
#
# Optional:
#   $env:JETSON_REPO = "~/HierarchalIDKCascades"
#   .\scripts\run_jetson_remote.ps1 -MaxSamples 500 -SyncOnly
#   .\scripts\run_jetson_remote.ps1 -SkipSync

param(
    [string]$JetsonHost = $env:JETSON_HOST,
    [string]$JetsonUser = $(if ($env:JETSON_USER) { $env:JETSON_USER } else { "jetson" }),
    [string]$RemoteRepo = $(if ($env:JETSON_REPO) { $env:JETSON_REPO } else { "~/HierarchalIDKCascades" }),
    [int]$MaxSamples = 500,
    [switch]$SyncOnly,
    [switch]$SkipSync
)

$ErrorActionPreference = "Stop"

if (-not $JetsonHost) {
    Write-Host @"
JETSON_HOST is not set.

1. Connect Jetson to same Wi‑Fi/Ethernet as this PC.
2. On Jetson, run:  hostname -I
3. On Windows:
     `$env:JETSON_HOST = "PASTE_IP_HERE"
     `$env:JETSON_USER = "jetson"
     .\scripts\run_jetson_remote.ps1

First-time only (copy entire project):
     scp -r C:\Users\sborg\IDKCascadeResearchCode\HierarchalIDKCascades ${JetsonUser}@YOUR_IP:~/

On Jetson (SSH in once):
     cd ~/HierarchalIDKCascades
     bash scripts/setup_jetson.sh
"@
    exit 1
}

$Target = "${JetsonUser}@${JetsonHost}"
$LocalRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

if (-not $SkipSync) {
    Write-Host "=== Syncing code + checkpoints to $Target ... ==="
    ssh $Target "mkdir -p $RemoteRepo"
    scp -r `
        "$LocalRoot\cascade" `
        "$LocalRoot\utils" `
        "$LocalRoot\training" `
        "$LocalRoot\models" `
        "$LocalRoot\profile_jetson.py" `
        "$LocalRoot\run_baselines.py" `
        "$LocalRoot\export_timing_latex.py" `
        "$LocalRoot\compare_cascade_timing.py" `
        "$LocalRoot\profile_probability_tables.py" `
        "$LocalRoot\scripts" `
        "${Target}:${RemoteRepo}/"
    scp -r "$LocalRoot\checkpoints" "${Target}:${RemoteRepo}/"
    scp -r "$LocalRoot\datasets" "${Target}:${RemoteRepo}/"
}

if ($SyncOnly) {
    Write-Host "Sync only — done."
    exit 0
}

Write-Host "=== Profiling on $Target (this may take 30–60 min) ... ==="
ssh $Target @"
cd $RemoteRepo
sudo jetson_clocks 2>/dev/null || true
python3 profile_jetson.py --max-samples $MaxSamples --kdet-mode model
"@

Write-Host "=== Pulling results back ... ==="
$Out = Join-Path $LocalRoot "checkpoints"
$Pull = @(
    "jetson_profile_report.json",
    "wcet_profile.json",
    "timing_comparison.json",
    "baseline_comparison_jetson.json",
    "cascade_pareto_sweep_jetson.json"
)
foreach ($f in $Pull) {
    scp "${Target}:${RemoteRepo}/checkpoints/$f" $Out 2>$null
}

Write-Host "=== Regenerating LaTeX/PNG tables ... ==="
Push-Location $LocalRoot
python export_timing_latex.py --format both
Pop-Location

Write-Host "Done. See checkpoints/jetson_profile_report.json"
