#include "solver3d.h"
#include <chrono>
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
static SolverProfile3D *active_profile = NULL;
static double profile_start_ms = 0.0;

typedef enum ProfileSection {
    PROFILE_SOURCE_ADD = 0,
    PROFILE_DIFFUSE,
    PROFILE_PROJECT,
    PROFILE_ADVECT,
    PROFILE_BOUNDARY,
    PROFILE_OBSTACLE,
    PROFILE_FADE
} ProfileSection;

static ProfileSection active_solve_section = PROFILE_DIFFUSE;

static double *profile_bucket(ProfileSection section) {
    if (!active_profile) return NULL;
    switch (section) {
    case PROFILE_SOURCE_ADD: return &active_profile->source_add_ms;
    case PROFILE_DIFFUSE: return &active_profile->diffuse_ms;
    case PROFILE_PROJECT: return &active_profile->project_ms;
    case PROFILE_ADVECT: return &active_profile->advect_ms;
    case PROFILE_BOUNDARY: return &active_profile->boundary_ms;
    case PROFILE_OBSTACLE: return &active_profile->obstacle_ms;
    case PROFILE_FADE: return &active_profile->fade_ms;
    default: return NULL;
    }
}

static double profile_now_ms(void) {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
        Clock::now().time_since_epoch()).count();
}

static void profile_begin_event(void) {
    if (!active_profile) return;
    profile_start_ms = profile_now_ms();
}

static void profile_end_event(ProfileSection section) {
    double *bucket = profile_bucket(section);
    if (!bucket) return;
    CUDA_CHECK(cudaDeviceSynchronize());
    *bucket += profile_now_ms() - profile_start_ms;
}

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

void solver3d_set_profile(SolverProfile3D *profile) {
    active_profile = profile;
}

__global__ void add_source_kernel3d(int N, float *__restrict__ x,
                                    const float *__restrict__ s, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < N + 2 && j < N + 2 && k < N + 2) {
        int idx = IX3(i, j, k);
        x[idx] += dt * s[idx];
    }
}

__global__ void add_velocity_source_kernel3d(int N,
                                             float *__restrict__ u,
                                             float *__restrict__ v,
                                             float *__restrict__ w,
                                             const float *__restrict__ u0,
                                             const float *__restrict__ v0,
                                             const float *__restrict__ w0,
                                             float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < N + 2 && j < N + 2 && k < N + 2) {
        int idx = IX3(i, j, k);
        u[idx] += dt * u0[idx];
        v[idx] += dt * v0[idx];
        w[idx] += dt * w0[idx];
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
    profile_begin_event();
    set_bnd_kernel3d<<<blocks, threads>>>(N, b, x);
    CUDA_CHECK(cudaPeekAtLastError());
    profile_end_event(PROFILE_BOUNDARY);
    int edge_blocks = (N + 255) / 256;
    profile_begin_event();
    set_bnd_edges_kernel3d<<<edge_blocks, 256>>>(N, x);
    CUDA_CHECK(cudaPeekAtLastError());
    profile_end_event(PROFILE_BOUNDARY);
    profile_begin_event();
    set_bnd_corners_kernel3d<<<1, 1>>>(N, x);
    CUDA_CHECK(cudaPeekAtLastError());
    profile_end_event(PROFILE_BOUNDARY);
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

    profile_begin_event();
    CUDA_CHECK(cudaMemcpy(d_tmp, x, bytes, cudaMemcpyDeviceToDevice));
    profile_end_event(active_solve_section);
    for (int iter = 0; iter < 20; iter++) {
        profile_begin_event();
        lin_solve_jacobi_kernel3d<<<blocks, threads>>>(N, write, read, x0, a, c);
        CUDA_CHECK(cudaPeekAtLastError());
        profile_end_event(active_solve_section);
        set_bnd3d(N, b, write);
        SWAP(read, write);
    }
    if (read != x) {
        profile_begin_event();
        CUDA_CHECK(cudaMemcpy(x, read, bytes, cudaMemcpyDeviceToDevice));
        profile_end_event(active_solve_section);
    }
}

static void diffuse3d(int N, int b, float *x, float *x0, float diff, float dt) {
    if (diff == 0.0f) {
        size_t bytes = (size_t)(N + 2) * (N + 2) * (N + 2) * sizeof(float);
        profile_begin_event();
        CUDA_CHECK(cudaMemcpy(x, x0, bytes, cudaMemcpyDeviceToDevice));
        profile_end_event(PROFILE_DIFFUSE);
        set_bnd3d(N, b, x);
        return;
    }
    float a = dt * diff * N * N;
    ProfileSection previous = active_solve_section;
    active_solve_section = PROFILE_DIFFUSE;
    lin_solve3d(N, b, x, x0, a, 1.0f + 6.0f * a);
    active_solve_section = previous;
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
    profile_begin_event();
    advect_kernel3d<<<blocks, threads>>>(N, b, d, d0, u, v, w, dt * N);
    CUDA_CHECK(cudaPeekAtLastError());
    profile_end_event(PROFILE_ADVECT);
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
    profile_begin_event();
    project_div_kernel3d<<<blocks, threads>>>(N, u, v, w, p, div);
    CUDA_CHECK(cudaPeekAtLastError());
    profile_end_event(PROFILE_PROJECT);
    set_bnd3d(N, 0, div);
    set_bnd3d(N, 0, p);
    ProfileSection previous = active_solve_section;
    active_solve_section = PROFILE_PROJECT;
    lin_solve3d(N, 0, p, div, 1.0f, 6.0f);
    active_solve_section = previous;
    profile_begin_event();
    project_update_kernel3d<<<blocks, threads>>>(N, u, v, w, p);
    CUDA_CHECK(cudaPeekAtLastError());
    profile_end_event(PROFILE_PROJECT);
    set_bnd3d(N, 1, u);
    set_bnd3d(N, 2, v);
    set_bnd3d(N, 3, w);
}

void dens_step3d(int N, float *x, float *x0,
                 float *u, float *v, float *w,
                 float diff, float dt) {
    dim3 threads(8, 8, 4);
    dim3 blocks((N + 2 + 7) / 8, (N + 2 + 7) / 8, (N + 2 + 3) / 4);
    profile_begin_event();
    add_source_kernel3d<<<blocks, threads>>>(N, x, x0, dt);
    CUDA_CHECK(cudaPeekAtLastError());
    profile_end_event(PROFILE_SOURCE_ADD);
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
    profile_begin_event();
    add_velocity_source_kernel3d<<<blocks, threads>>>(N, u, v, w, u0, v0, w0, dt);
    CUDA_CHECK(cudaPeekAtLastError());
    profile_end_event(PROFILE_SOURCE_ADD);
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
    profile_begin_event();
    fade_fields_kernel3d<<<blocks, threads>>>(N, dens, u, v, w,
                                              dissipation, vel_damping);
    CUDA_CHECK(cudaPeekAtLastError());
    profile_end_event(PROFILE_FADE);
}

__global__ void sphere_obstacle_kernel3d(int N,
                                         float *__restrict__ dens,
                                         float *__restrict__ u,
                                         float *__restrict__ v,
                                         float *__restrict__ w,
                                         float cx, float cy, float cz,
                                         float radius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 1 || i > N || j < 1 || j > N || k < 1 || k > N) return;

    float dx = (float)i - cx;
    float dy = (float)j - cy;
    float dz = (float)k - cz;
    float dist2 = dx * dx + dy * dy + dz * dz;
    float r2 = radius * radius;
    int idx = IX3(i, j, k);

    if (dist2 <= r2) {
        dens[idx] = 0.0f;
        u[idx] = 0.0f;
        v[idx] = 0.0f;
        w[idx] = 0.0f;
        return;
    }

    float shell = radius + 3.0f;
    if (dist2 <= shell * shell) {
        float inv_dist = rsqrtf(dist2);
        float nx = dx * inv_dist;
        float ny = dy * inv_dist;
        float nz = dz * inv_dist;
        float inward = u[idx] * nx + v[idx] * ny + w[idx] * nz;
        if (inward < 0.0f) {
            u[idx] -= inward * nx;
            v[idx] -= inward * ny;
            w[idx] -= inward * nz;
        }
    }
}

void apply_sphere_obstacle3d(int N, float *dens, float *u, float *v, float *w,
                             float cx, float cy, float cz, float radius) {
    dim3 threads(8, 8, 4);
    dim3 blocks((N + 2 + 7) / 8, (N + 2 + 7) / 8, (N + 2 + 3) / 4);
    profile_begin_event();
    sphere_obstacle_kernel3d<<<blocks, threads>>>(N, dens, u, v, w,
                                                  cx, cy, cz, radius);
    CUDA_CHECK(cudaPeekAtLastError());
    profile_end_event(PROFILE_OBSTACLE);
}
