param()

$ErrorActionPreference = "Stop"

function Quote-Arg($arg) {
    if ($arg -match '[\s"]') {
        return '"' + ($arg -replace '"', '\"') + '"'
    }
    return $arg
}

$cl = (Get-Command cl.exe -ErrorAction SilentlyContinue).Source
$vcvars = $null
if (-not $cl) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vsInstall) {
            $candidate = Join-Path $vsInstall "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $candidate) {
                $vcvars = $candidate
            }
        }
    }
    if (-not $vcvars) {
        throw "cl.exe was not found. Install Visual Studio Build Tools with the Desktop development with C++ workload, or run this script from a Developer PowerShell."
    }
}

$args = @(
    "/O2", "/EHsc", "/std:c++17", "/utf-8", "/wd4828",
    "main.cpp",
    "/Fe:fluid3d_cpu.exe"
)

Write-Host "cl $($args -join ' ')"
if ($vcvars) {
    $quotedArgs = (($args | ForEach-Object { Quote-Arg $_ }) -join " ")
    $cmd = '"' + $vcvars + '" && cl ' + $quotedArgs
    cmd.exe /d /c $cmd
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} else {
    & $cl @args
}

Write-Host ""
Write-Host "Build complete."
Write-Host "Run benchmark: .\fluid3d_cpu.exe --benchmark"
