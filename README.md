# Fluid CUDA

2D fluid simulation experiments based on Jos Stam's **Stable Fluids** method from
"Real-Time Fluid Dynamics for Games" (GDC 2003).

The repository contains separate CPU and CUDA GPU viewers so each version can be
compiled, executed, and profiled independently.

## Theoretical Background

This project simulates smoke-like fluid motion on a regular 2D grid. The visible
smoke is represented by a scalar **density field**, and the flow is represented
by a 2D **velocity field** `(u, v)`.

The solver follows the incompressible Navier-Stokes model:

```text
velocity change = external force + diffusion + advection + projection
density change  = source injection + diffusion + advection + dissipation
```

The main steps are:

| Step | Meaning |
|------|---------|
| Source | Add smoke density or velocity from the mouse/automatic emitter |
| Diffusion | Spread density or velocity to nearby cells |
| Advection | Move density/velocity along the current velocity field |
| Projection | Remove divergence so the velocity field behaves incompressibly |
| Dissipation | Gradually fade smoke and damp velocity |

The key idea of Stable Fluids is that it prioritizes visual stability over strict
physical exactness. Instead of using an unstable forward particle-style update,
it traces values backward through the velocity field during advection. This is
called **semi-Lagrangian advection**, and it lets the simulation run with larger
time steps without exploding numerically.

For pressure projection, the solver uses iterative Jacobi relaxation. This makes
the velocity field approximately divergence-free, which prevents the fluid from
visually expanding or compressing like a gas being created from nowhere.

## CPU vs GPU Version

The CPU version performs the full simulation with ordinary C++ loops. It is useful
as a readable baseline.

The GPU version moves the simulation kernels to CUDA. Each grid cell can be
updated in parallel, which makes advection, diffusion, projection, and damping
good candidates for GPU acceleration.

Current GPU-side optimizations include:

- skipping diffusion solves when `diff` or `visc` is zero
- rendering density as one OpenGL texture instead of drawing `N * N` quads
- skipping velocity `DeviceToHost` readback unless velocity overlay is enabled

## Repository Layout

| Path | Purpose |
|------|---------|
| `cpu/` | CPU-only GLUT viewer |
| `gpu/` | CUDA GPU-only GLUT viewer |
| `gpu/solver.cu` | CUDA solver kernels |
| `gpu/solver.h` | CUDA solver declarations |
| `gpu/THREE_D_PLAN.md` | Roadmap for extending the GPU version to 3D smoke |

## Requirements

Windows:

- NVIDIA CUDA Toolkit
- Visual Studio Build Tools with the **Desktop development with C++** workload
- freeglut

If you use Anaconda, freeglut can be installed with:

```powershell
conda install -c conda-forge freeglut
```

The build scripts look for freeglut under:

```text
%USERPROFILE%\anaconda3\Library
```

## Build And Run

Clone the repository:

```powershell
git clone https://github.com/Do1Yun/Fluid_CUDA.git
cd Fluid_CUDA
```

Build and run the CPU version:

```powershell
cd cpu
powershell -ExecutionPolicy Bypass -File .\build.ps1
.\fluid2d_cpu.exe
```

Build and run the GPU version:

```powershell
cd ..\gpu
powershell -ExecutionPolicy Bypass -File .\build.ps1
.\fluid2d_gpu.exe
```

## Controls

| Input | Action |
|-------|--------|
| Left mouse drag | Add velocity |
| Right mouse drag | Add smoke density |
| `c` | Clear simulation |
| `p` | Pause/resume |
| `v` | Toggle velocity-field overlay |
| `q` or `ESC` | Quit |

## Notes

The default grid size is controlled by `SIZE` in each viewer:

- `cpu/main.cpp`
- `gpu/main.cu`

Larger values create more detailed smoke but increase simulation and rendering
cost. The GPU version is the main target for future 3D expansion.
