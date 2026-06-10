param(
    [ValidateSet("2d", "3d", "all")]
    [string]$Task = "all",
    [int]$Size2D = 256,
    [int]$Frames2D = 240,
    [int]$Warmup2D = 30,
    [int]$Size3D = 32,
    [int]$Frames3D = 120,
    [int]$Warmup3D = 20,
    [switch]$Build
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Invoke-InDir($dir, $exe, $argsList) {
    Push-Location (Join-Path $root $dir)
    try {
        Write-Host "[$dir] $exe $($argsList -join ' ')"
        & ".\$exe" @argsList
        if ($LASTEXITCODE -ne 0) {
            throw "$exe failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

function Build-InDir($dir) {
    Push-Location (Join-Path $root $dir)
    try {
        powershell -ExecutionPolicy Bypass -File .\build.ps1
        if ($LASTEXITCODE -ne 0) {
            throw "$dir build failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

if ($Build) {
    if ($Task -eq "2d" -or $Task -eq "all") {
        Build-InDir "cpu"
        Build-InDir "gpu"
    }
    if ($Task -eq "3d" -or $Task -eq "all") {
        Build-InDir "cpu3d"
        Build-InDir "gpu3d"
    }
}

if ($Task -eq "2d" -or $Task -eq "all") {
    $args2d = @("--benchmark", "--bench-size", "$Size2D", "--bench-warmup", "$Warmup2D", "--bench-frames", "$Frames2D")
    Invoke-InDir "cpu" "fluid2d_cpu.exe" $args2d
    Invoke-InDir "gpu" "fluid2d_gpu.exe" $args2d
}

if ($Task -eq "3d" -or $Task -eq "all") {
    $args3d = @("--benchmark", "--bench-size", "$Size3D", "--bench-warmup", "$Warmup3D", "--bench-frames", "$Frames3D")
    Invoke-InDir "cpu3d" "fluid3d_cpu.exe" $args3d
    Invoke-InDir "gpu3d" "fluid3d_gpu.exe" $args3d
}

Write-Host ""
Write-Host "2D CSV: benchmark_results\benchmark_2d.csv"
Write-Host "3D CSV: benchmark_results\benchmark_3d.csv"
