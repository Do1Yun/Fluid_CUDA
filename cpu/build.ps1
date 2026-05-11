param()

$ErrorActionPreference = "Stop"

function Quote-Arg($arg) {
    if ($arg -match '[\s"]') {
        return '"' + ($arg -replace '"', '\"') + '"'
    }
    return $arg
}

$condaRoot = Join-Path $env:USERPROFILE "anaconda3"
$glutRoot = Join-Path $condaRoot "Library"
$glutInclude = Join-Path $glutRoot "include"
$glutLib = Join-Path $glutRoot "lib"
$glutDll = Join-Path $glutRoot "bin\freeglut.dll"

if (-not (Test-Path (Join-Path $glutInclude "GL\glut.h"))) {
    throw "Could not find GL\glut.h. Install freeglut, for example with: conda install -c conda-forge freeglut"
}
if (-not (Test-Path (Join-Path $glutLib "freeglut.lib"))) {
    throw "Could not find freeglut.lib. Install freeglut, for example with: conda install -c conda-forge freeglut"
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
    "/I$glutInclude",
    "main.cpp",
    "/Fe:fluid2d_cpu.exe",
    "/link",
    "/LIBPATH:$glutLib",
    "freeglut.lib", "opengl32.lib", "glu32.lib"
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

if (Test-Path $glutDll) {
    Copy-Item $glutDll -Destination ".\freeglut.dll" -Force
}

Write-Host ""
Write-Host "Build complete."
Write-Host "Run viewer: .\fluid2d_cpu.exe"
