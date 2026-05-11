# Fluid CUDA

2D Stable Fluids experiments based on Jos Stam's "Real-Time Fluid Dynamics for Games" (GDC 2003).

The project is split into separate CPU and CUDA GPU viewers so they can be profiled independently.

## Layout

| Path | Purpose |
|------|---------|
| `cpu/` | CPU-only GLUT viewer |
| `gpu/` | CUDA GPU-only GLUT viewer |
| `gpu/THREE_D_PLAN.md` | Plan for extending the GPU version to 3D smoke |

## Windows Dependencies

Install:
- NVIDIA CUDA Toolkit for the GPU version
- Visual Studio Build Tools with the Desktop development with C++ workload
- freeglut

With Anaconda, freeglut can be installed with:

```powershell
conda install -c conda-forge freeglut
```

## Build And Run

CPU:

```powershell
cd cpu
powershell -ExecutionPolicy Bypass -File .\build.ps1
.\fluid2d_cpu.exe
```

GPU:

```powershell
cd gpu
powershell -ExecutionPolicy Bypass -File .\build.ps1
.\fluid2d_gpu.exe
```

## Controls

| Input | Action |
|-------|--------|
| Left mouse drag | Add velocity |
| Right mouse drag | Add density |
| `c` | Clear simulation |
| `p` | Pause/resume |
| `v` | Toggle velocity-field overlay |
| `q` or `ESC` | Quit |

## GPU Notes

The GPU viewer currently uses CUDA for simulation and uploads the density field as an OpenGL texture for display. This avoids the old immediate-mode path that drew one quad per simulation cell.

Current GPU optimization milestones:
- Skip diffusion solves when `diff` or `visc` is zero.
- Render density through one texture quad instead of `N * N` OpenGL quads.
- Skip velocity `DeviceToHost` readback unless the velocity overlay is enabled.
- Next: move source injection to CUDA kernels and remove full source-buffer host uploads.
