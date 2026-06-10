param(
    [ValidateSet("2d", "3d", "all")]
    [string]$Task = "all",
    [string]$Sizes2D = "64,128,256,512,1024,2048",
    [string]$Sizes3D = "16,32,64,128,256",
    [int]$Repeats = 10,
    [int]$Frames2D = 10,
    [int]$Warmup2D = 3,
    [int]$Frames3D = 5,
    [int]$Warmup3D = 2,
    [int]$CPUTimeoutSec2D = 8,
    [int]$CPUTimeoutSec3D = 20,
    [int]$GPUTimeoutSec = 180,
    [switch]$Build,
    [switch]$Reset
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$resultDir = Join-Path $root "benchmark_results"
$culture = [System.Globalization.CultureInfo]::InvariantCulture

function Parse-Sizes($text) {
    return $text.Split(",") | ForEach-Object { [int]$_.Trim() } | Where-Object { $_ -gt 0 }
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

function Quote-ProcessArg($arg) {
    if ($arg -match '[\s"]') {
        return '"' + ($arg -replace '"', '\"') + '"'
    }
    return $arg
}

function Format-Value($value) {
    if ($null -eq $value) { return "NaN" }
    if ($value -is [double] -and [double]::IsNaN($value)) { return "NaN" }
    if ($value -is [single] -and [single]::IsNaN($value)) { return "NaN" }
    if ($value -is [int] -or $value -is [long]) { return "$value" }
    if ($value -is [double] -or $value -is [single] -or $value -is [decimal]) {
        return ([double]$value).ToString("0.#####", $culture)
    }
    return [string]$value
}

function New-NanRow($task, $mode, $inputId, $n, $dimension, $cells, $fieldMb, $warmup, $frames, $repeat, $timeout) {
    return [pscustomobject]@{
        row_type = "raw"
        timestamp = (Get-Date -Format "yyyyMMdd_HHmmss")
        task = $task
        mode = $mode
        input_id = $inputId
        N = $n
        dimension = $dimension
        cells = $cells
        scalar_field_mb = $fieldMb
        warmup = $warmup
        frames = $frames
        repeat = $repeat
        source_ms = [double]::NaN
        step_ms = [double]::NaN
        total_ms = [double]::NaN
        fps = [double]::NaN
        mcells_per_sec = [double]::NaN
        ns_per_cell = [double]::NaN
        density_sum = [double]::NaN
        density_max = [double]::NaN
        velocity_l2 = [double]::NaN
        divergence_l2 = [double]::NaN
        divergence_max = [double]::NaN
        speedup_vs_cpu = [double]::NaN
        timeout = $timeout
    }
}

function Invoke-BenchmarkProcess($dir, $exe, $argsList, $timeoutSec, $task, $mode, $inputId, $n, $dimension, $warmup, $frames, $repeat) {
    $workDir = Join-Path $root $dir
    $cells = if ($dimension -eq 2) { [int64]$n * [int64]$n } else { [int64]$n * [int64]$n * [int64]$n }
    $fieldMb = if ($dimension -eq 2) {
        [double](($n + 2) * ($n + 2) * 4) / (1024.0 * 1024.0)
    } else {
        [double](($n + 2) * ($n + 2) * ($n + 2) * 4) / (1024.0 * 1024.0)
    }

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = Join-Path $workDir $exe
    $psi.Arguments = (($argsList | ForEach-Object { Quote-ProcessArg $_ }) -join " ")
    $psi.WorkingDirectory = $workDir
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $psi
    [void]$process.Start()
    $finished = $process.WaitForExit($timeoutSec * 1000)
    if (-not $finished) {
        try { $process.Kill($true) } catch { try { $process.Kill() } catch {} }
        $process.WaitForExit()
        return New-NanRow $task $mode $inputId $n $dimension $cells $fieldMb $warmup $frames $repeat 1
    }

    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    if ($process.ExitCode -ne 0) {
        Write-Warning "$mode N=$n repeat=$repeat failed: $stderr"
        return New-NanRow $task $mode $inputId $n $dimension $cells $fieldMb $warmup $frames $repeat 1
    }

    $line = ($stdout -split "`r?`n" | Where-Object { $_ -like "benchmark_result,*" } | Select-Object -Last 1)
    if (-not $line) {
        Write-Warning "$mode N=$n repeat=$repeat produced no benchmark_result"
        return New-NanRow $task $mode $inputId $n $dimension $cells $fieldMb $warmup $frames $repeat 1
    }

    $parts = $line.Split(",")
    $sourceMs = [double]::Parse($parts[5], $culture)
    $stepMs = [double]::Parse($parts[6], $culture)
    $totalMs = [double]::Parse($parts[7], $culture)
    $fps = [double]::Parse($parts[8], $culture)
    $densitySum = [double]::Parse($parts[9], $culture)
    $densityMax = [double]::Parse($parts[10], $culture)
    $velocityL2 = [double]::Parse($parts[11], $culture)
    $divergenceL2 = [double]::Parse($parts[12], $culture)
    $divergenceMax = [double]::Parse($parts[13], $culture)
    $mcells = if ($totalMs -gt 0.0) { ([double]$cells / ($totalMs / 1000.0)) / 1000000.0 } else { [double]::NaN }
    $nsPerCell = if ($cells -gt 0) { $totalMs * 1000000.0 / [double]$cells } else { [double]::NaN }

    return [pscustomobject]@{
        row_type = "raw"
        timestamp = (Get-Date -Format "yyyyMMdd_HHmmss")
        task = $task
        mode = $mode
        input_id = $inputId
        N = $n
        dimension = $dimension
        cells = $cells
        scalar_field_mb = $fieldMb
        warmup = $warmup
        frames = $frames
        repeat = $repeat
        source_ms = $sourceMs
        step_ms = $stepMs
        total_ms = $totalMs
        fps = $fps
        mcells_per_sec = $mcells
        ns_per_cell = $nsPerCell
        density_sum = $densitySum
        density_max = $densityMax
        velocity_l2 = $velocityL2
        divergence_l2 = $divergenceL2
        divergence_max = $divergenceMax
        speedup_vs_cpu = [double]::NaN
        timeout = 0
    }
}

function Average-Field($rows, $field) {
    $valid = @($rows | Where-Object { $_.timeout -eq 0 -and -not [double]::IsNaN([double]$_.($field)) })
    if ($valid.Count -eq 0) { return [double]::NaN }
    return (($valid | Measure-Object -Property $field -Average).Average)
}

function New-SummaryRow($rows, $mode, $cpuTotalMs) {
    $first = $rows[0]
    $anyTimeout = @($rows | Where-Object { $_.timeout -ne 0 }).Count -gt 0
    $validCount = @($rows | Where-Object { $_.timeout -eq 0 }).Count
    $totalMs = if ($anyTimeout -or $validCount -eq 0) { [double]::NaN } else { Average-Field $rows "total_ms" }
    $speedup = [double]::NaN
    if (-not [double]::IsNaN($totalMs) -and -not [double]::IsNaN($cpuTotalMs) -and $totalMs -gt 0.0) {
        $speedup = $cpuTotalMs / $totalMs
    }
    return [pscustomobject]@{
        row_type = "summary_avg"
        timestamp = (Get-Date -Format "yyyyMMdd_HHmmss")
        task = $first.task
        mode = $mode
        input_id = $first.input_id
        N = $first.N
        dimension = $first.dimension
        cells = $first.cells
        scalar_field_mb = $first.scalar_field_mb
        warmup = $first.warmup
        frames = $first.frames
        repeat = "avg_$($rows.Count)"
        source_ms = if ($anyTimeout) { [double]::NaN } else { Average-Field $rows "source_ms" }
        step_ms = if ($anyTimeout) { [double]::NaN } else { Average-Field $rows "step_ms" }
        total_ms = $totalMs
        fps = if ([double]::IsNaN($totalMs) -or $totalMs -le 0.0) { [double]::NaN } else { 1000.0 / $totalMs }
        mcells_per_sec = if ([double]::IsNaN($totalMs) -or $totalMs -le 0.0) { [double]::NaN } else { ([double]$first.cells / ($totalMs / 1000.0)) / 1000000.0 }
        ns_per_cell = if ([double]::IsNaN($totalMs)) { [double]::NaN } else { $totalMs * 1000000.0 / [double]$first.cells }
        density_sum = if ($anyTimeout) { [double]::NaN } else { Average-Field $rows "density_sum" }
        density_max = if ($anyTimeout) { [double]::NaN } else { Average-Field $rows "density_max" }
        velocity_l2 = if ($anyTimeout) { [double]::NaN } else { Average-Field $rows "velocity_l2" }
        divergence_l2 = if ($anyTimeout) { [double]::NaN } else { Average-Field $rows "divergence_l2" }
        divergence_max = if ($anyTimeout) { [double]::NaN } else { Average-Field $rows "divergence_max" }
        speedup_vs_cpu = $speedup
        timeout = if ($anyTimeout) { 1 } else { 0 }
    }
}

function Write-Rows($path, $rows, $append) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $path) | Out-Null
    if (-not $append -and (Test-Path $path)) { Remove-Item $path -Force }
    $header = "row_type,timestamp,task,mode,input_id,N,dimension,cells,scalar_field_mb,warmup,frames,repeat,source_ms,step_ms,total_ms,fps,mcells_per_sec,ns_per_cell,density_sum,density_max,velocity_l2,divergence_l2,divergence_max,speedup_vs_cpu,timeout"
    if (-not (Test-Path $path)) { Add-Content -Path $path -Value $header }
    foreach ($r in $rows) {
        $values = @(
            $r.row_type, $r.timestamp, $r.task, $r.mode, $r.input_id, $r.N,
            $r.dimension, $r.cells, (Format-Value $r.scalar_field_mb), $r.warmup,
            $r.frames, $r.repeat, (Format-Value $r.source_ms),
            (Format-Value $r.step_ms), (Format-Value $r.total_ms),
            (Format-Value $r.fps), (Format-Value $r.mcells_per_sec),
            (Format-Value $r.ns_per_cell), (Format-Value $r.density_sum),
            (Format-Value $r.density_max), (Format-Value $r.velocity_l2),
            (Format-Value $r.divergence_l2), (Format-Value $r.divergence_max),
            (Format-Value $r.speedup_vs_cpu), $r.timeout
        )
        Add-Content -Path $path -Value ($values -join ",")
    }
}

function Run-Task($taskName, $sizes, $modes, $frames, $warmup, $csvPath) {
    $taskRawRows = @()
    $taskSummaryRows = @()
    foreach ($n in $sizes) {
        $allRowsForSize = @()
        $summaryRows = @()
        $cpuSummaryTotal = [double]::NaN
        $modeRows = @{}
        foreach ($mode in $modes) {
            $rows = @()
            $timedOutForSize = $false
            for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
                $isCpu = $mode.Mode -like "cpu*"
                $cells = if ($mode.Dimension -eq 2) { [int64]$n * [int64]$n } else { [int64]$n * [int64]$n * [int64]$n }
                $fieldMb = if ($mode.Dimension -eq 2) {
                    [double](($n + 2) * ($n + 2) * 4) / (1024.0 * 1024.0)
                } else {
                    [double](($n + 2) * ($n + 2) * ($n + 2) * 4) / (1024.0 * 1024.0)
                }
                if ($isCpu -and $timedOutForSize) {
                    $row = New-NanRow $taskName $mode.Mode $mode.InputId $n $mode.Dimension $cells $fieldMb $warmup $frames $repeat 1
                } else {
                    $timeout = if ($isCpu) {
                        if ($mode.Dimension -eq 2) { $CPUTimeoutSec2D } else { $CPUTimeoutSec3D }
                    } else {
                        $GPUTimeoutSec
                    }
                    $args = @("--benchmark", "--bench-no-csv", "--bench-tag", "scaling", "--bench-size", "$n", "--bench-warmup", "$warmup", "--bench-frames", "$frames")
                    Write-Host "[$($mode.Mode)] N=$n repeat=$repeat/$Repeats timeout=${timeout}s"
                    $row = Invoke-BenchmarkProcess $mode.Dir $mode.Exe $args $timeout $taskName $mode.Mode $mode.InputId $n $mode.Dimension $warmup $frames $repeat
                    if ($isCpu -and $row.timeout -ne 0) { $timedOutForSize = $true }
                }
                $rows += $row
                $allRowsForSize += $row
            }
            $modeRows[$mode.Mode] = $rows
            if ($mode.Mode -like "cpu*") {
                $cpuSummary = New-SummaryRow $rows $mode.Mode ([double]::NaN)
                $cpuSummary.speedup_vs_cpu = if ($cpuSummary.timeout -eq 0) { 1.0 } else { [double]::NaN }
                $cpuSummaryTotal = [double]$cpuSummary.total_ms
                $summaryRows += $cpuSummary
            }
        }
        foreach ($mode in $modes) {
            if ($mode.Mode -like "gpu*") {
                $summaryRows += New-SummaryRow $modeRows[$mode.Mode] $mode.Mode $cpuSummaryTotal
            }
        }
        $taskRawRows += $allRowsForSize
        $taskSummaryRows += $summaryRows
    }
    Write-Rows $csvPath ($taskRawRows + $taskSummaryRows) $true
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

New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
if ($Reset) {
    if ($Task -eq "2d" -or $Task -eq "all") { Remove-Item (Join-Path $resultDir "benchmark_scaling_2d.csv") -ErrorAction SilentlyContinue }
    if ($Task -eq "3d" -or $Task -eq "all") { Remove-Item (Join-Path $resultDir "benchmark_scaling_3d.csv") -ErrorAction SilentlyContinue }
}

if ($Task -eq "2d" -or $Task -eq "all") {
    $modes2d = @(
        [pscustomobject]@{ Mode = "cpu2d"; Dir = "cpu"; Exe = "fluid2d_cpu.exe"; Dimension = 2; InputId = "deterministic_2d_v1" },
        [pscustomobject]@{ Mode = "gpu2d"; Dir = "gpu"; Exe = "fluid2d_gpu.exe"; Dimension = 2; InputId = "deterministic_2d_v1" }
    )
    Run-Task "2d" (Parse-Sizes $Sizes2D) $modes2d $Frames2D $Warmup2D (Join-Path $resultDir "benchmark_scaling_2d.csv")
}

if ($Task -eq "3d" -or $Task -eq "all") {
    $modes3d = @(
        [pscustomobject]@{ Mode = "cpu3d"; Dir = "cpu3d"; Exe = "fluid3d_cpu.exe"; Dimension = 3; InputId = "deterministic_3d_wind_tunnel_v1" },
        [pscustomobject]@{ Mode = "gpu3d"; Dir = "gpu3d"; Exe = "fluid3d_gpu.exe"; Dimension = 3; InputId = "deterministic_3d_wind_tunnel_v1" }
    )
    Run-Task "3d" (Parse-Sizes $Sizes3D) $modes3d $Frames3D $Warmup3D (Join-Path $resultDir "benchmark_scaling_3d.csv")
}

Write-Host ""
Write-Host "2D scaling CSV: benchmark_results\benchmark_scaling_2d.csv"
Write-Host "3D scaling CSV: benchmark_results\benchmark_scaling_3d.csv"
