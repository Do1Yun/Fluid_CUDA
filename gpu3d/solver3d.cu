#include "solver3d.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define IX3(i, j, k) ((i) + (N + 2) * ((j) + (N + 2) * (k)))
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

static float *d_tmp = NULL;

void init_solver3d(int n) {
    size_t size = (size_t)(n + 2) * (n + 2) * (n + 2) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_tmp, size));
}

void free_solver3d() {
    if (d_tmp) {
        CUDA_CHECK(cudaFree(d_tmp));
        d_tmp = NULL;
    }
}

__global__ void add_source_kernel3d(int N, float *x, float *s, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < N + 2 && j < N + 2 && k < N + 2) {
        int idx = IX3(i, j, k);
        x[idx] += dt * s[idx];
    }
}

__global__ void set_bnd_kernel3d(int N, int b, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= N && j <= N) {
        x[IX3(0, i, j)]     = (b == 1) ? -x[IX3(1, i, j)] : x[IX3(1, i, j)];
        x[IX3(N + 1, i, j)] = (b == 1) ? -x[IX3(N, i, j)] : x[IX3(N, i, j)];
        x[IX3(i, 0, j)]     = (b == 2) ? -x[IX3(i, 1, j)] : x[IX3(i, 1, j)];
        x[IX3(i, N + 1, j)] = (b == 2) ? -x[IX3(i, N, j)] : x[IX3(i, N, j)];
        x[IX3(i, j, 0)]     = (b == 3) ? -x[IX3(i, j, 1)] : x[IX3(i, j, 1)];
        x[IX3(i, j, N + 1)] = (b == 3) ? -x[IX3(i, j, N)] : x[IX3(i, j, N)];
    }
}

__global__ void set_bnd_corners_kernel3d(int N, float *x) {
    x[IX3(0, 0, 0)] =
        (x[IX3(1, 0, 0)] + x[IX3(0, 1, 0)] + x[IX3(0, 0, 1)]) / 3.0f;
    x[IX3(0, 0, N + 1)] =
        (x[IX3(1, 0, N + 1)] + x[IX3(0, 1, N + 1)] + x[IX3(0, 0, N)]) / 3.0f;
    x[IX3(0, N + 1, 0)] =
        (x[IX3(1, N + 1, 0)] + x[IX3(0, N, 0)] + x[IX3(0, N + 1, 1)]) / 3.0f;
    x[IX3(0, N + 1, N + 1)] =
        (x[IX3(1, N + 1, N + 1)] + x[IX3(0, N, N + 1)] + x[IX3(0, N + 1, N)]) / 3.0f;
    x[IX3(N + 1, 0, 0)] =
        (x[IX3(N, 0, 0)] + x[IX3(N + 1, 1, 0)] + x[IX3(N + 1, 0, 1)]) / 3.0f;
    x[IX3(N + 1, 0, N + 1)] =
        (x[IX3(N, 0, N + 1)] + x[IX3(N + 1, 1, N + 1)] + x[IX3(N + 1, 0, N)]) / 3.0f;
    x[IX3(N + 1, N + 1, 0)] =
        (x[IX3(N, N + 1, 0)] + x[IX3(N + 1, N, 0)] + x[IX3(N + 1, N + 1, 1)]) / 3.0f;
    x[IX3(N + 1, N + 1, N + 1)] =
        (x[IX3(N, N + 1, N + 1)] + x[IX3(N + 1, N, N + 1)] + x[IX3(N + 1, N + 1, N)]) / 3.0f;
}

__global__ void set_bnd_edges_kernel3d(int N, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i <= N) {
        x[IX3(0, 0, i)] =
            0.5f * (x[IX3(1, 0, i)] + x[IX3(0, 1, i)]);
        x[IX3(0, N + 1, i)] =
            0.5f * (x[IX3(1, N + 1, i)] + x[IX3(0, N, i)]);
        x[IX3(N + 1, 0, i)] =
            0.5f * (x[IX3(N, 0, i)] + x[IX3(N + 1, 1, i)]);
        x[IX3(N + 1, N + 1, i)] =
            0.5f * (x[IX3(N, N + 1, i)] + x[IX3(N + 1, N, i)]);

        x[IX3(0, i, 0)] =
            0.5f * (x[IX3(1, i, 0)] + x[IX3(0, i, 1)]);
        x[IX3(0, i, N + 1)] =
            0.5f * (x[IX3(1, i, N + 1)] + x[IX3(0, i, N)]);
        x[IX3(N + 1, i, 0)] =
            0.5f * (x[IX3(N, i, 0)] + x[IX3(N + 1, i, 1)]);
        x[IX3(N + 1, i, N + 1)] =
            0.5f * (x[IX3(N, i, N + 1)] + x[IX3(N + 1, i, N)]);

        x[IX3(i, 0, 0)] =
            0.5f * (x[IX3(i, 1, 0)] + x[IX3(i, 0, 1)]);
        x[IX3(i, 0, N + 1)] =
            0.5f * (x[IX3(i, 1, N + 1)] + x[IX3(i, 0, N)]);
        x[IX3(i, N + 1, 0)] =
            0.5f * (x[IX3(i, N, 0)] + x[IX3(i, N + 1, 1)]);
        x[IX3(i, N + 1, N + 1)] =
            0.5f * (x[IX3(i, N, N + 1)] + x[IX3(i, N + 1, N)]);
    }
}

static void set_bnd3d(int N, int b, float *x) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    set_bnd_kernel3d<<<blocks, threads>>>(N, b, x);
    CUDA_CHECK(cudaPeekAtLastError());
    int edge_blocks = (N + 255) / 256;
    set_bnd_edges_kernel3d<<<edge_blocks, 256>>>(N, x);
    CUDA_CHECK(cudaPeekAtLastError());
    set_bnd_corners_kernel3d<<<1, 1>>>(N, x);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void lin_solve_jacobi_kernel3d(int N, float *x_new, float *x,
                                          float *x0, float a, float c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= 1 && i <= N && j >= 1 && j <= N && k >= 1 && k <= N) {
        x_new[IX3(i, j, k)] =
            (x0[IX3(i, j, k)] +
             a * (x[IX3(i - 1, j, k)] + x[IX3(i + 1, j, k)] +
                  x[IX3(i, j - 1, k)] + x[IX3(i, j + 1, k)] +
                  x[IX3(i, j, k - 1)] + x[IX3(i, j, k + 1)])) / c;
    }
}

static void lin_solve3d(int N, int b, float *x, float *x0, float a, float c) {
    dim3 threads(8, 8, 4);
    dim3 blocks((N + 2 + 7) / 8, (N + 2 + 7) / 8, (N + 2 + 3) / 4);
    size_t bytes = (size_t)(N + 2) * (N + 2) * (N + 2) * sizeof(float);
    float *read = x;
    float *write = d_tmp;

    CUDA_CHECK(cudaMemcpy(d_tmp, x, bytes, cudaMemcpyDeviceToDevice));
    for (int iter = 0; iter < 20; iter++) {
        lin_solve_jacobi_kernel3d<<<blocks, threads>>>(N, write, read, x0, a, c);
        CUDA_CHECK(cudaPeekAtLastError());
        set_bnd3d(N, b, write);
        SWAP(read, write);
    }
    if (read != x) {
        CUDA_CHECK(cudaMemcpy(x, read, bytes, cudaMemcpyDeviceToDevice));
    }
}

static void diffuse3d(int N, int b, float *x, float *x0, float diff, float dt) {
    if (diff == 0.0f) {
        size_t bytes = (size_t)(N + 2) * (N + 2) * (N + 2) * sizeof(float);
        CUDA_CHECK(cudaMemcpy(x, x0, bytes, cudaMemcpyDeviceToDevice));
        set_bnd3d(N, b, x);
        return;
    }
    float a = dt * diff * N * N;
    lin_solve3d(N, b, x, x0, a, 1.0f + 6.0f * a);
}

__global__ void advect_kernel3d(int N, int b, float *d, float *d0,
                                float *u, float *v, float *w, float dt0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= 1 && i <= N && j >= 1 && j <= N && k >= 1 && k <= N) {
        float x = i - dt0 * u[IX3(i, j, k)];
        float y = j - dt0 * v[IX3(i, j, k)];
        float z = k - dt0 * w[IX3(i, j, k)];
        if (x < 0.5f) x = 0.5f;
        if (x > N + 0.5f) x = N + 0.5f;
        if (y < 0.5f) y = 0.5f;
        if (y > N + 0.5f) y = N + 0.5f;
        if (z < 0.5f) z = 0.5f;
        if (z > N + 0.5f) z = N + 0.5f;

        int i0 = (int)x, j0 = (int)y, k0 = (int)z;
        int i1 = i0 + 1, j1 = j0 + 1, k1 = k0 + 1;
        float s1 = x - i0, s0 = 1.0f - s1;
        float t1 = y - j0, t0 = 1.0f - t1;
        float r1 = z - k0, r0 = 1.0f - r1;

        d[IX3(i, j, k)] =
            s0 * (t0 * (r0 * d0[IX3(i0, j0, k0)] + r1 * d0[IX3(i0, j0, k1)]) +
                  t1 * (r0 * d0[IX3(i0, j1, k0)] + r1 * d0[IX3(i0, j1, k1)])) +
            s1 * (t0 * (r0 * d0[IX3(i1, j0, k0)] + r1 * d0[IX3(i1, j0, k1)]) +
                  t1 * (r0 * d0[IX3(i1, j1, k0)] + r1 * d0[IX3(i1, j1, k1)]));
    }
}

static void advect3d(int N, int b, float *d, float *d0,
                     float *u, float *v, float *w, float dt) {
    dim3 threads(8, 8, 4);
    dim3 blocks((N + 2 + 7) / 8, (N + 2 + 7) / 8, (N + 2 + 3) / 4);
    advect_kernel3d<<<blocks, threads>>>(N, b, d, d0, u, v, w, dt * N);
    CUDA_CHECK(cudaPeekAtLastError());
    set_bnd3d(N, b, d);
}

__global__ void project_div_kernel3d(int N, float *u, float *v, float *w,
                                     float *p, float *div) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= 1 && i <= N && j >= 1 && j <= N && k >= 1 && k <= N) {
        div[IX3(i, j, k)] =
            -0.5f * (u[IX3(i + 1, j, k)] - u[IX3(i - 1, j, k)] +
                     v[IX3(i, j + 1, k)] - v[IX3(i, j - 1, k)] +
                     w[IX3(i, j, k + 1)] - w[IX3(i, j, k - 1)]) / N;
        p[IX3(i, j, k)] = 0.0f;
    }
}

__global__ void project_update_kernel3d(int N, float *u, float *v, float *w,
                                        float *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= 1 && i <= N && j >= 1 && j <= N && k >= 1 && k <= N) {
        u[IX3(i, j, k)] -= 0.5f * N * (p[IX3(i + 1, j, k)] - p[IX3(i - 1, j, k)]);
        v[IX3(i, j, k)] -= 0.5f * N * (p[IX3(i, j + 1, k)] - p[IX3(i, j - 1, k)]);
        w[IX3(i, j, k)] -= 0.5f * N * (p[IX3(i, j, k + 1)] - p[IX3(i, j, k - 1)]);
    }
}

static void project3d(int N, float *u, float *v, float *w, float *p, float *div) {
    dim3 threads(8, 8, 4);
    dim3 blocks((N + 2 + 7) / 8, (N + 2 + 7) / 8, (N + 2 + 3) / 4);
    project_div_kernel3d<<<blocks, threads>>>(N, u, v, w, p, div);
    CUDA_CHECK(cudaPeekAtLastError());
    set_bnd3d(N, 0, div);
    set_bnd3d(N, 0, p);
    lin_solve3d(N, 0, p, div, 1.0f, 6.0f);
    project_update_kernel3d<<<blocks, threads>>>(N, u, v, w, p);
    CUDA_CHECK(cudaPeekAtLastError());
    set_bnd3d(N, 1, u);
    set_bnd3d(N, 2, v);
    set_bnd3d(N, 3, w);
}

void dens_step3d(int N, float *x, float *x0,
                 float *u, float *v, float *w,
                 float diff, float dt) {
    dim3 threads(8, 8, 4);
    dim3 blocks((N + 2 + 7) / 8, (N + 2 + 7) / 8, (N + 2 + 3) / 4);
    add_source_kernel3d<<<blocks, threads>>>(N, x, x0, dt);
    CUDA_CHECK(cudaPeekAtLastError());
    SWAP(x0, x);
    diffuse3d(N, 0, x, x0, diff, dt);
    SWAP(x0, x);
    advect3d(N, 0, x, x0, u, v, w, dt);
}

void vel_step3d(int N, float *u, float *v, float *w,
                float *u0, float *v0, float *w0,
                float visc, float dt) {
    dim3 threads(8, 8, 4);
    dim3 blocks((N + 2 + 7) / 8, (N + 2 + 7) / 8, (N + 2 + 3) / 4);
    add_source_kernel3d<<<blocks, threads>>>(N, u, u0, dt);
    add_source_kernel3d<<<blocks, threads>>>(N, v, v0, dt);
    add_source_kernel3d<<<blocks, threads>>>(N, w, w0, dt);
    CUDA_CHECK(cudaPeekAtLastError());
    SWAP(u0, u);
    diffuse3d(N, 1, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse3d(N, 2, v, v0, visc, dt);
    SWAP(w0, w);
    diffuse3d(N, 3, w, w0, visc, dt);
    project3d(N, u, v, w, u0, v0);
    SWAP(u0, u);
    SWAP(v0, v);
    SWAP(w0, w);
    advect3d(N, 1, u, u0, u0, v0, w0, dt);
    advect3d(N, 2, v, v0, u0, v0, w0, dt);
    advect3d(N, 3, w, w0, u0, v0, w0, dt);
    project3d(N, u, v, w, u0, v0);
}

__global__ void fade_fields_kernel3d(int N, float *dens,
                                     float *u, float *v, float *w,
                                     float dissipation, float vel_damping) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < N + 2 && j < N + 2 && k < N + 2) {
        int idx = IX3(i, j, k);
        dens[idx] *= dissipation;
        u[idx] *= vel_damping;
        v[idx] *= vel_damping;
        w[idx] *= vel_damping;
    }
}

void fade_fields3d(int N, float *dens, float *u, float *v, float *w,
                   float dissipation, float vel_damping) {
    dim3 threads(8, 8, 4);
    dim3 blocks((N + 2 + 7) / 8, (N + 2 + 7) / 8, (N + 2 + 3) / 4);
    fade_fields_kernel3d<<<blocks, threads>>>(N, dens, u, v, w,
                                              dissipation, vel_damping);
    CUDA_CHECK(cudaPeekAtLastError());
}
