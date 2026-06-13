from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


NUMERIC_COLUMNS = [
    "N",
    "cells",
    "scalar_field_mb",
    "warmup",
    "frames",
    "source_ms",
    "step_ms",
    "total_ms",
    "fps",
    "mcells_per_sec",
    "ns_per_cell",
    "density_sum",
    "density_max",
    "velocity_l2",
    "divergence_l2",
    "divergence_max",
    "source_add_ms",
    "diffuse_ms",
    "project_ms",
    "advect_ms",
    "boundary_ms",
    "obstacle_ms",
    "fade_ms",
    "other_ms",
    "speedup_vs_cpu",
    "timeout",
]

PROFILE_COLUMNS = [
    ("source_add_ms", "source"),
    ("diffuse_ms", "diffuse"),
    ("project_ms", "project"),
    ("advect_ms", "advect"),
    ("boundary_ms", "boundary"),
    ("obstacle_ms", "obstacle"),
    ("fade_ms", "fade"),
    ("other_ms", "other"),
]

PROFILE_COLORS = {
    "source_add_ms": "#4c78a8",
    "diffuse_ms": "#72b7b2",
    "project_ms": "#f58518",
    "advect_ms": "#54a24b",
    "boundary_ms": "#e45756",
    "obstacle_ms": "#b279a2",
    "fade_ms": "#ff9da6",
    "other_ms": "#bab0ac",
}


def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, na_values=["NaN", "NAN", ""])
    df = df[df["row_type"] == "summary_avg"].copy()
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values(["N", "mode"])


def clean_number(value: float) -> str:
    if pd.isna(value):
        return "NaN"
    text = f"{value:.5f}".rstrip("0").rstrip(".")
    return text if text else "0"


def plot_task(summary: pd.DataFrame, task_label: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    colors = {
        "cpu2d": "#4c78a8",
        "gpu2d": "#f58518",
        "cpu3d": "#4c78a8",
        "gpu3d": "#f58518",
    }
    markers = {"cpu2d": "o", "gpu2d": "s", "cpu3d": "o", "gpu3d": "s"}

    for mode, group in summary.groupby("mode", sort=True):
        group = group.sort_values("N")
        label = mode.upper()
        axes[0].plot(
            group["N"],
            group["total_ms"],
            marker=markers.get(mode, "o"),
            linewidth=2.2,
            label=label,
            color=colors.get(mode),
        )
        axes[1].plot(
            group["N"],
            group["mcells_per_sec"],
            marker=markers.get(mode, "o"),
            linewidth=2.2,
            label=label,
            color=colors.get(mode),
        )

    gpu = summary[summary["mode"].str.startswith("gpu")].sort_values("N")
    if not gpu.empty:
        axes[2].plot(
            gpu["N"],
            gpu["speedup_vs_cpu"],
            marker="D",
            linewidth=2.2,
            label="GPU / CPU",
            color="#54a24b",
        )

    axes[0].set_title(f"{task_label} total time")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("total_ms (lower is better)")
    axes[0].set_yscale("log")

    axes[1].set_title(f"{task_label} throughput")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("mcells_per_sec (higher is better)")
    axes[1].set_yscale("log")

    axes[2].set_title(f"{task_label} GPU speedup")
    axes[2].set_xlabel("N")
    axes[2].set_ylabel("speedup_vs_cpu (higher is better)")
    axes[2].axhline(1.0, color="#999999", linestyle="--", linewidth=1.2)

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        ax.set_xticks(sorted(summary["N"].dropna().unique()))
        ax.xaxis.set_major_formatter(ScalarFormatter())

    fig.suptitle(f"Stable Fluid {task_label} Scaling Benchmark", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_profile_breakdown(summary: pd.DataFrame, task_label: str, out_path: Path) -> bool:
    columns = [column for column, _label in PROFILE_COLUMNS]
    if any(column not in summary.columns for column in columns):
        return False
    if summary[columns].notna().sum().sum() == 0:
        return False

    modes = sorted(summary["mode"].dropna().unique())
    if not modes:
        return False

    fig, axes = plt.subplots(
        len(modes),
        1,
        figsize=(12.5, max(4.2, 3.1 * len(modes))),
        constrained_layout=True,
        squeeze=False,
    )

    for ax, mode in zip(axes[:, 0], modes):
        group = summary[summary["mode"] == mode].sort_values("N")
        x = np.arange(len(group))
        bottom = np.zeros(len(group), dtype=float)

        for column, label in PROFILE_COLUMNS:
            values = group[column].fillna(0.0).to_numpy(dtype=float)
            if np.all(values == 0.0):
                continue
            ax.bar(
                x,
                values,
                bottom=bottom,
                label=label,
                color=PROFILE_COLORS.get(column),
                width=0.72,
            )
            bottom += values

        ax.plot(
            x,
            group["step_ms"].to_numpy(dtype=float),
            color="#222222",
            marker="o",
            linewidth=1.5,
            label="step_ms",
        )
        ax.set_title(mode.upper())
        ax.set_ylabel("ms per frame")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(n)) for n in group["N"]])
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=4)

    axes[-1, 0].set_xlabel("N")
    fig.suptitle(f"Stable Fluid {task_label} Solver Pass Breakdown", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def make_summary_table(summary_2d: pd.DataFrame, summary_3d: pd.DataFrame, out_path: Path) -> None:
    rows: list[list[str]] = []
    for task_label, summary, cpu_mode, gpu_mode in [
        ("2D", summary_2d, "cpu2d", "gpu2d"),
        ("3D", summary_3d, "cpu3d", "gpu3d"),
    ]:
        for n in sorted(summary["N"].dropna().unique()):
            cpu = summary[(summary["N"] == n) & (summary["mode"] == cpu_mode)]
            gpu = summary[(summary["N"] == n) & (summary["mode"] == gpu_mode)]
            cpu_row = cpu.iloc[0] if not cpu.empty else None
            gpu_row = gpu.iloc[0] if not gpu.empty else None
            rows.append(
                [
                    task_label,
                    str(int(n)),
                    clean_number(cpu_row["total_ms"]) if cpu_row is not None else "NaN",
                    clean_number(gpu_row["total_ms"]) if gpu_row is not None else "NaN",
                    clean_number(gpu_row["speedup_vs_cpu"]) if gpu_row is not None else "NaN",
                    clean_number(gpu_row["mcells_per_sec"]) if gpu_row is not None else "NaN",
                    str(int(cpu_row["timeout"])) if cpu_row is not None and not pd.isna(cpu_row["timeout"]) else "NaN",
                ]
            )

    columns = [
        "Task",
        "N",
        "CPU total_ms",
        "GPU total_ms",
        "GPU speedup",
        "GPU Mcells/s",
        "CPU timeout",
    ]
    fig_height = max(4.5, 0.42 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.35)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e9eef5")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fb")
    ax.set_title("Stable Fluid Scaling Summary", fontsize=14, pad=18)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stable Fluid benchmark CSV files.")
    parser.add_argument("--input-dir", default="benchmark_results", help="Directory containing benchmark CSV files.")
    parser.add_argument("--output-dir", default="benchmark_results", help="Directory for generated PNG files.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_2d = load_summary(input_dir / "benchmark_scaling_2d.csv")
    summary_3d = load_summary(input_dir / "benchmark_scaling_3d.csv")

    outputs = [
        output_dir / "benchmark_scaling_2d_performance.png",
        output_dir / "benchmark_scaling_3d_performance.png",
        output_dir / "benchmark_scaling_summary_table.png",
    ]
    plot_task(summary_2d, "2D", outputs[0])
    plot_task(summary_3d, "3D", outputs[1])
    make_summary_table(summary_2d, summary_3d, outputs[2])

    profile_2d = output_dir / "benchmark_scaling_2d_profile_breakdown.png"
    profile_3d = output_dir / "benchmark_scaling_3d_profile_breakdown.png"
    if plot_profile_breakdown(summary_2d, "2D", profile_2d):
        outputs.append(profile_2d)
    if plot_profile_breakdown(summary_3d, "3D", profile_3d):
        outputs.append(profile_3d)

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
