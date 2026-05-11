#include "solver.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define IX(i, j) ((i) + (N + 2) * (j))
#define SWAP(x0, x) { float *tmp = x0; x0 = x; x = tmp; }

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA ERROR %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err__));          \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

static float *d_x_new = NULL;

void init_solver(int N) {
    int size = (N + 2) * (N + 2) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_x_new, size));
}

void free_solver() {
    if (d_x_new) {
        CUDA_CHECK(cudaFree(d_x_new));
        d_x_new = NULL;
    }
}

__global__ void add_source_kernel(int N, float *x, float *s, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N + 2 && j < N + 2) {
        int idx = IX(i, j);
        x[idx] += dt * s[idx];
    }
}

__global__ void set_bnd_kernel(int N, int b, float *x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (idx <= N) {
        x[IX(0, idx)]     = (b == 1) ? -x[IX(1, idx)] : x[IX(1, idx)];
        x[IX(N + 1, idx)] = (b == 1) ? -x[IX(N, idx)] : x[IX(N, idx)];
        x[IX(idx, 0)]     = (b == 2) ? -x[IX(idx, 1)] : x[IX(idx, 1)];
        x[IX(idx, N + 1)] = (b == 2) ? -x[IX(idx, N)] : x[IX(idx, N)];
    }
}

__global__ void set_bnd_corners_kernel(int N, float *x) {
    x[IX(0, 0)]         = 0.5f * (x[IX(1, 0)]     + x[IX(0, 1)]);
    x[IX(0, N + 1)]     = 0.5f * (x[IX(1, N + 1)] + x[IX(0, N)]);
    x[IX(N + 1, 0)]     = 0.5f * (x[IX(N, 0)]     + x[IX(N + 1, 1)]);
    x[IX(N + 1, N + 1)] = 0.5f * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

static void set_bnd(int N, int b, float *d_x) {
    int blocks = (N + 255) / 256;
    set_bnd_kernel<<<blocks, 256>>>(N, b, d_x);
    CUDA_CHECK(cudaPeekAtLastError());
    set_bnd_corners_kernel<<<1, 1>>>(N, d_x);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void lin_solve_jacobi_kernel(int N, float *x_new, float *x, float *x0, float a, float c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i <= N && j >= 1 && j <= N) {
        x_new[IX(i, j)] = (x0[IX(i, j)] +
                           a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                                x[IX(i, j - 1)] + x[IX(i, j + 1)])) / c;
    }
}

static void lin_solve(int N, int b, float *d_x, float *d_x0, float a, float c) {
    dim3 threads(16, 16);
    dim3 blocks((N + 2 + 15) / 16, (N + 2 + 15) / 16);
    size_t bytes = (size_t)(N + 2) * (N + 2) * sizeof(float);
    float *read = d_x;
    float *write = d_x_new;

    CUDA_CHECK(cudaMemcpy(d_x_new, d_x, bytes, cudaMemcpyDeviceToDevice));

    for (int k = 0; k < 20; k++) {
        lin_solve_jacobi_kernel<<<blocks, threads>>>(N, write, read, d_x0, a, c);
        CUDA_CHECK(cudaPeekAtLastError());
        set_bnd(N, b, write);
        SWAP(read, write);
    }

    if (read != d_x) {
        CUDA_CHECK(cudaMemcpy(d_x, read, bytes, cudaMemcpyDeviceToDevice));
    }
}

static void diffuse(int N, int b, float *x, float *x0, float diff, float dt) {
    if (diff == 0.0f) {
        size_t bytes = (size_t)(N + 2) * (N + 2) * sizeof(float);
        CUDA_CHECK(cudaMemcpy(x, x0, bytes, cudaMemcpyDeviceToDevice));
        set_bnd(N, b, x);
        return;
    }

    float a = dt * diff * N * N;
    lin_solve(N, b, x, x0, a, 1 + 4 * a);
}

__global__ void advect_kernel(int N, int b, float *d, float *d0, float *u, float *v, float dt0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i <= N && j >= 1 && j <= N) {
        float x = i - dt0 * u[IX(i, j)];
        float y = j - dt0 * v[IX(i, j)];
        if (x < 0.5f) x = 0.5f;
        if (x > N + 0.5f) x = N + 0.5f;
        int i0 = (int)x;
        int i1 = i0 + 1;
        if (y < 0.5f) y = 0.5f;
        if (y > N + 0.5f) y = N + 0.5f;
        int j0 = (int)y;
        int j1 = j0 + 1;
        float s1 = x - i0;
        float s0 = 1.0f - s1;
        float t1 = y - j0;
        float t0 = 1.0f - t1;
        d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                      s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    }
}

static void advect(int N, int b, float *d, float *d0, float *u, float *v, float dt) {
    dim3 threads(16, 16);
    dim3 blocks((N + 2 + 15) / 16, (N + 2 + 15) / 16);
    float dt0 = dt * N;
    advect_kernel<<<blocks, threads>>>(N, b, d, d0, u, v, dt0);
    CUDA_CHECK(cudaPeekAtLastError());
    set_bnd(N, b, d);
}

__global__ void project_div_kernel(int N, float *u, float *v, float *p, float *div) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i <= N && j >= 1 && j <= N) {
        div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] +
                                 v[IX(i, j + 1)] - v[IX(i, j - 1)]) / N;
        p[IX(i, j)] = 0;
    }
}

__global__ void project_update_kernel(int N, float *u, float *v, float *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i <= N && j >= 1 && j <= N) {
        u[IX(i, j)] -= 0.5f * N * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
        v[IX(i, j)] -= 0.5f * N * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
    }
}

static void project(int N, float *u, float *v, float *p, float *div) {
    dim3 threads(16, 16);
    dim3 blocks((N + 2 + 15) / 16, (N + 2 + 15) / 16);
    project_div_kernel<<<blocks, threads>>>(N, u, v, p, div);
    CUDA_CHECK(cudaPeekAtLastError());
    set_bnd(N, 0, div);
    set_bnd(N, 0, p);
    lin_solve(N, 0, p, div, 1, 4);
    project_update_kernel<<<blocks, threads>>>(N, u, v, p);
    CUDA_CHECK(cudaPeekAtLastError());
    set_bnd(N, 1, u);
    set_bnd(N, 2, v);
}

void dens_step(int N, float *x, float *x0, float *u, float *v, float diff, float dt) {
    dim3 threads(16, 16);
    dim3 blocks((N + 2 + 15) / 16, (N + 2 + 15) / 16);
    add_source_kernel<<<blocks, threads>>>(N, x, x0, dt);
    CUDA_CHECK(cudaPeekAtLastError());
    SWAP(x0, x);
    diffuse(N, 0, x, x0, diff, dt);
    SWAP(x0, x);
    advect(N, 0, x, x0, u, v, dt);
}

void vel_step(int N, float *u, float *v, float *u0, float *v0, float visc, float dt) {
    dim3 threads(16, 16);
    dim3 blocks((N + 2 + 15) / 16, (N + 2 + 15) / 16);
    add_source_kernel<<<blocks, threads>>>(N, u, u0, dt);
    CUDA_CHECK(cudaPeekAtLastError());
    add_source_kernel<<<blocks, threads>>>(N, v, v0, dt);
    CUDA_CHECK(cudaPeekAtLastError());
    SWAP(u0, u);
    diffuse(N, 1, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse(N, 2, v, v0, visc, dt);
    project(N, u, v, u0, v0);
    SWAP(u0, u);
    SWAP(v0, v);
    advect(N, 1, u, u0, u0, v0, dt);
    advect(N, 2, v, v0, u0, v0, dt);
    project(N, u, v, u0, v0);
}

__global__ void fade_fields_kernel(int N, float *d_dens, float *d_u, float *d_v, float dissipation, float vel_damping) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N + 2 && j < N + 2) {
        int idx = IX(i, j);
        d_dens[idx] *= dissipation;
        d_u[idx] *= vel_damping;
        d_v[idx] *= vel_damping;
    }
}

void fade_fields(int N, float *d_dens, float *d_u, float *d_v, float dissipation, float vel_damping) {
    dim3 threads(16, 16);
    dim3 blocks((N + 2 + 15) / 16, (N + 2 + 15) / 16);
    fade_fields_kernel<<<blocks, threads>>>(N, d_dens, d_u, d_v, dissipation, vel_damping);
    CUDA_CHECK(cudaPeekAtLastError());
}
