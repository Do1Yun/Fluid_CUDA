# GPU 3D Fluid Plan

Goal: extend the CUDA viewer from a 2D stable-fluids solver to a GPU-first 3D smoke volume without bringing CPU simulation back into the render loop.

## Current 2D Upgrade

- 2D CPU/GPU viewers now render smoke color from velocity magnitude.
- Slow smoke maps toward blue, faster smoke shifts through the spectrum toward red.
- Keep this color rule as the first debug view for 3D slices and ray-marched volumes.

## Phase 1 - Split and Baseline

- Keep `gpu/` as the only CUDA simulation path.
- Keep rendering single-window and GPU-only so timing is not polluted by the CPU comparison viewer.
- Add the 3D version as a separate executable, for example `fluid3d_gpu.exe`, instead of merging modes into `main.cu`.
- Add a headless 3D CUDA smoke test before adding the 3D viewer.

## Phase 2 - 3D Solver Core

- Change fields from `(N + 2)^2` to `(Nx + 2) * (Ny + 2) * (Nz + 2)`.
- Store velocity as `u`, `v`, `w`; density as a scalar volume.
- Port kernels in this order: add source, boundary, Jacobi linear solve, advection, projection, dissipation.
- Use 3D CUDA launch grids and flatten with `IX(i, j, k)`.
- Start with small volumes like `64^3`, then tune toward `128^3`.
- Treat `128^3` and above as GPU-first territory; avoid CPU reference stepping except for tiny regression tests like `16^3`.

## Phase 3 - GPU Rendering

- Replace immediate-mode quad drawing with a texture or buffer path.
- For first 3D display, render volume slices to verify correctness.
- Then add ray-marched volume rendering through a 3D texture.
- Avoid CPU readback per frame once the 3D solver is stable; use CUDA/OpenGL interop for density upload.

## Phase 4 - Interaction

- Map mouse input to a 3D injection point using camera ray plus depth plane.
- Match common Unreal/Unity editor navigation:
  - `W/A/S/D`: move forward/left/back/right.
  - `Q/E`: move down/up.
  - Hold right mouse: free-look camera yaw/pitch.
  - Mouse wheel: adjust movement speed or zoom in slice mode.
  - `F`: focus camera on the smoke source or selected injection point.
- Add controls for slice view, volume view, density source strength, dissipation, and camera orbit/free-look.
- Keep CPU-side UI state small; simulation buffers stay on device.

## Phase 5 - Optimization

- Use ping-pong device buffers instead of host copies.
- Fuse simple kernels where it reduces launch overhead.
- Profile Jacobi iteration count and consider red-black Gauss-Seidel, multigrid, or FFT pressure solve later.
- Keep render resolution independent from simulation resolution.
