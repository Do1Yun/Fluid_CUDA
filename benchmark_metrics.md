# Benchmark Metrics

Benchmark CSV files are written under `benchmark_results/`.

## Files

- `benchmark_2d.csv`: 2D task results. CPU 2D and GPU 2D append rows here.
- `benchmark_3d.csv`: 3D task results. CPU 3D and GPU 3D append rows here.
- `benchmark_scaling_2d.csv`: 2D scaling results across multiple `N` values.
- `benchmark_scaling_3d.csv`: 3D scaling results across multiple `N` values.

## Columns

| Column | Meaning | Direction |
|---|---|---|
| `row_type` | `raw` for each repeat, `summary_avg` for the per-mode/per-size average. | Reference only |
| `timestamp` | Local run time, `YYYYMMDD_HHMMSS`. | Reference only |
| `task` | `2d` or `3d`. | Reference only |
| `mode` | Implementation, such as `cpu2d`, `gpu2d`, `cpu3d`, `gpu3d`. | Reference only |
| `input_id` | Deterministic benchmark input scenario. | Reference only |
| `N` | Interior grid resolution. 2D is `N*N`; 3D is `N*N*N`. | Compare only at same task/input unless testing scaling |
| `dimension` | 2 or 3. | Reference only |
| `cells` | Interior cell count. | Higher means larger workload |
| `scalar_field_mb` | Memory size of one scalar field including ghost cells. | Lower is lighter memory pressure |
| `warmup` | Frames run before timing accumulation. | Reference only |
| `frames` | Timed frames. | Higher gives more stable averages |
| `repeat` | Repeat index for raw rows, or `avg_N` for summary rows. | Reference only |
| `source_ms` | Average deterministic source setup time per frame. | Lower is better |
| `step_ms` | Average solver step time per frame. | Lower is better |
| `total_ms` | `source_ms + step_ms`. | Lower is better |
| `fps` | `1000 / total_ms`. | Higher is better |
| `mcells_per_sec` | Interior cells processed per second, in millions, based on `total_ms`. | Higher is better |
| `ns_per_cell` | Average nanoseconds per interior cell, based on `total_ms`. | Lower is better |
| `density_sum` | Total density remaining after the run. | Scenario dependent |
| `density_max` | Maximum density value. | Scenario dependent |
| `velocity_l2` | RMS velocity magnitude. | Scenario dependent |
| `divergence_l2` | RMS divergence after projection. | Lower is better |
| `divergence_max` | Maximum absolute divergence after projection. | Lower is better |
| `speedup_vs_cpu` | CPU `total_ms` divided by this row's `total_ms`; present on summary rows. | Higher is better; `1` is CPU baseline; `NaN` if CPU timed out |
| `timeout` | `1` if the run or summary contains an early timeout. | Lower is better |

## Notes

- Use `step_ms`, `total_ms`, `mcells_per_sec`, and `ns_per_cell` for performance comparisons.
- Use `divergence_l2` and `divergence_max` for projection quality checks.
- `density_sum`, `density_max`, and `velocity_l2` are not simple better/worse scores. They are regression signals: compare them against previous runs with the same task, input, grid size, frame count, and solver settings.
- CPU and GPU solvers may not produce identical density/divergence values because the CPU paths use in-place iterative solves while GPU paths use Jacobi-style ping-pong solves.
- Very small `N` values are usually CPU-favorable because GPU kernel launch, synchronization, and host/device transfer overhead dominate the actual stencil work. Use the scaling CSV files to find the crossover range.

## Scaling Runs

Use the scaling runner to evaluate performance over multiple grid sizes. It runs each size/mode multiple times, writes raw rows, then appends summary average rows:

```powershell
.\run_scaling_benchmarks.ps1 -Task 2d -Reset
.\run_scaling_benchmarks.ps1 -Task 3d -Reset
```

Default scaling sizes:

- 2D: `64,128,256,512,1024,2048`
- 3D: `16,32,64,128,256`

Default repeat count:

- `Repeats=10`

Default CPU timeout:

- 2D CPU: 8 seconds per repeat
- 3D CPU: 20 seconds per repeat

When a CPU run times out for a given `N`, the runner stops launching more CPU repeats for that size and fills the remaining raw rows plus the CPU summary row with `NaN`.

You can override them:

```powershell
.\run_scaling_benchmarks.ps1 -Task 2d -Sizes2D "128,256,512,1024,1536"
.\run_scaling_benchmarks.ps1 -Task 3d -Sizes3D "24,32,48,64,96"
```

To inspect the averages only:

```powershell
Import-Csv benchmark_results\benchmark_scaling_2d.csv | Where-Object row_type -eq summary_avg
Import-Csv benchmark_results\benchmark_scaling_3d.csv | Where-Object row_type -eq summary_avg
```
