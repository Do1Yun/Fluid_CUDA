// 3D Stable Fluids - CPU benchmark-only runner
// Uses the same deterministic wind-tunnel benchmark scenario as gpu3d.

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <wchar.h>
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <sys/stat.h>
#endif

static int N = 32;
static float dt = 0.08f;
static float diff = 0.0f;
static float visc = 0.0f;
static float dissipation = 0.997f;
static float source = 420.0f;
static float domain_half_extent = 1.45f;
static float auto_flow_speed = 1.15f;
static float obstacle_radius_world = 0.28f;
static float obstacle_x = 0.0f, obstacle_y = 0.0f, obstacle_z = 0.0f;
static float wind_source_radius_world = 0.38f;
static float wind_source_depth_world = 0.30f;

static float *dens, *dens_prev;
static float *u, *v, *w, *u_prev, *v_prev, *w_prev;

typedef struct BenchmarkConfig {
    int enabled;
    int frames;
    int warmup;
    int grid_size;
    char tag[32];
    int save_csv;
} BenchmarkConfig;

#define IX3(i, j, k) ((i) + (N + 2) * ((j) + (N + 2) * (k)))
#define SWAP(x0, x) { float *tmp = x0; x0 = x; x = tmp; }

static double now_seconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}

static size_t volume_count(void) {
    return (size_t)(N + 2) * (N + 2) * (N + 2);
}

static float domain_size(void) {
    return domain_half_extent * 2.0f;
}

static float clamp_float(float value, float lo, float hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static float obstacle_radius_cells(void) {
    return obstacle_radius_world / domain_size() * (float)N;
}

static float world_to_cell_coord(float p) {
    return ((p + domain_half_extent) / domain_size()) * (float)N + 1.0f;
}

static int world_to_cell_index(float p) {
    int cell = (int)(((p + domain_half_extent) / domain_size()) * (float)N) + 1;
    if (cell < 1) cell = 1;
    if (cell > N) cell = N;
    return cell;
}

static float cell_to_world_coord(int index) {
    float t = ((float)index - 0.5f) / (float)N;
    return -domain_half_extent + t * domain_size();
}

static void clamp_obstacle_position(void) {
    float limit = domain_half_extent - obstacle_radius_world;
    obstacle_x = clamp_float(obstacle_x, -limit, limit);
    obstacle_y = clamp_float(obstacle_y, -limit, limit);
    obstacle_z = clamp_float(obstacle_z, -limit, limit);
}

static void wind_source_frame(float center[3], float dir[3],
                              float tangent[3], float bitangent[3]) {
    float half_depth = wind_source_depth_world * 0.5f;
    center[0] = 0.0f;
    center[1] = domain_half_extent - half_depth;
    center[2] = 0.0f;

    dir[0] = 0.0f;
    dir[1] = -1.0f;
    dir[2] = 0.0f;

    tangent[0] = 1.0f;
    tangent[1] = 0.0f;
    tangent[2] = 0.0f;

    bitangent[0] = 0.0f;
    bitangent[1] = 0.0f;
    bitangent[2] = 1.0f;
}

static int allocate_data(void) {
    size_t bytes = volume_count() * sizeof(float);
    dens = (float *)malloc(bytes);
    dens_prev = (float *)malloc(bytes);
    u = (float *)malloc(bytes);
    v = (float *)malloc(bytes);
    w = (float *)malloc(bytes);
    u_prev = (float *)malloc(bytes);
    v_prev = (float *)malloc(bytes);
    w_prev = (float *)malloc(bytes);
    if (!dens || !dens_prev || !u || !v || !w || !u_prev || !v_prev || !w_prev) {
        fprintf(stderr, "ERROR: out of memory\n");
        return 0;
    }
    return 1;
}

static void free_data(void) {
    free(dens); free(dens_prev);
    free(u); free(v); free(w);
    free(u_prev); free(v_prev); free(w_prev);
}

static void clear_host_sources(void) {
    size_t bytes = volume_count() * sizeof(float);
    memset(dens_prev, 0, bytes);
    memset(u_prev, 0, bytes);
    memset(v_prev, 0, bytes);
    memset(w_prev, 0, bytes);
}

static void clear_data(void) {
    size_t bytes = volume_count() * sizeof(float);
    memset(dens, 0, bytes);
    memset(u, 0, bytes);
    memset(v, 0, bytes);
    memset(w, 0, bytes);
    clear_host_sources();
}

static void set_bnd3d_cpu(int b, float *x) {
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            x[IX3(0, i, j)]     = (b == 1) ? -x[IX3(1, i, j)] : x[IX3(1, i, j)];
            x[IX3(N + 1, i, j)] = (b == 1) ? -x[IX3(N, i, j)] : x[IX3(N, i, j)];
            x[IX3(i, 0, j)]     = (b == 2) ? -x[IX3(i, 1, j)] : x[IX3(i, 1, j)];
            x[IX3(i, N + 1, j)] = (b == 2) ? -x[IX3(i, N, j)] : x[IX3(i, N, j)];
            x[IX3(i, j, 0)]     = (b == 3) ? -x[IX3(i, j, 1)] : x[IX3(i, j, 1)];
            x[IX3(i, j, N + 1)] = (b == 3) ? -x[IX3(i, j, N)] : x[IX3(i, j, N)];
        }
    }

    for (int i = 1; i <= N; i++) {
        x[IX3(0, 0, i)] = 0.5f * (x[IX3(1, 0, i)] + x[IX3(0, 1, i)]);
        x[IX3(0, N + 1, i)] = 0.5f * (x[IX3(1, N + 1, i)] + x[IX3(0, N, i)]);
        x[IX3(N + 1, 0, i)] = 0.5f * (x[IX3(N, 0, i)] + x[IX3(N + 1, 1, i)]);
        x[IX3(N + 1, N + 1, i)] = 0.5f * (x[IX3(N, N + 1, i)] + x[IX3(N + 1, N, i)]);

        x[IX3(0, i, 0)] = 0.5f * (x[IX3(1, i, 0)] + x[IX3(0, i, 1)]);
        x[IX3(0, i, N + 1)] = 0.5f * (x[IX3(1, i, N + 1)] + x[IX3(0, i, N)]);
        x[IX3(N + 1, i, 0)] = 0.5f * (x[IX3(N, i, 0)] + x[IX3(N + 1, i, 1)]);
        x[IX3(N + 1, i, N + 1)] = 0.5f * (x[IX3(N, i, N + 1)] + x[IX3(N + 1, i, N)]);

        x[IX3(i, 0, 0)] = 0.5f * (x[IX3(i, 1, 0)] + x[IX3(i, 0, 1)]);
        x[IX3(i, 0, N + 1)] = 0.5f * (x[IX3(i, 1, N + 1)] + x[IX3(i, 0, N)]);
        x[IX3(i, N + 1, 0)] = 0.5f * (x[IX3(i, N, 0)] + x[IX3(i, N + 1, 1)]);
        x[IX3(i, N + 1, N + 1)] = 0.5f * (x[IX3(i, N, N + 1)] + x[IX3(i, N + 1, N)]);
    }

    x[IX3(0, 0, 0)] = (x[IX3(1, 0, 0)] + x[IX3(0, 1, 0)] + x[IX3(0, 0, 1)]) / 3.0f;
    x[IX3(0, 0, N + 1)] = (x[IX3(1, 0, N + 1)] + x[IX3(0, 1, N + 1)] + x[IX3(0, 0, N)]) / 3.0f;
    x[IX3(0, N + 1, 0)] = (x[IX3(1, N + 1, 0)] + x[IX3(0, N, 0)] + x[IX3(0, N + 1, 1)]) / 3.0f;
    x[IX3(0, N + 1, N + 1)] = (x[IX3(1, N + 1, N + 1)] + x[IX3(0, N, N + 1)] + x[IX3(0, N + 1, N)]) / 3.0f;
    x[IX3(N + 1, 0, 0)] = (x[IX3(N, 0, 0)] + x[IX3(N + 1, 1, 0)] + x[IX3(N + 1, 0, 1)]) / 3.0f;
    x[IX3(N + 1, 0, N + 1)] = (x[IX3(N, 0, N + 1)] + x[IX3(N + 1, 1, N + 1)] + x[IX3(N + 1, 0, N)]) / 3.0f;
    x[IX3(N + 1, N + 1, 0)] = (x[IX3(N, N + 1, 0)] + x[IX3(N + 1, N, 0)] + x[IX3(N + 1, N + 1, 1)]) / 3.0f;
    x[IX3(N + 1, N + 1, N + 1)] = (x[IX3(N, N + 1, N + 1)] + x[IX3(N + 1, N, N + 1)] + x[IX3(N + 1, N + 1, N)]) / 3.0f;
}

static void lin_solve3d_cpu(int b, float *x, const float *x0, float a, float c) {
    for (int iter = 0; iter < 20; iter++) {
        for (int k = 1; k <= N; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1; i <= N; i++) {
                    x[IX3(i, j, k)] =
                        (x0[IX3(i, j, k)] +
                         a * (x[IX3(i - 1, j, k)] + x[IX3(i + 1, j, k)] +
                              x[IX3(i, j - 1, k)] + x[IX3(i, j + 1, k)] +
                              x[IX3(i, j, k - 1)] + x[IX3(i, j, k + 1)])) / c;
                }
            }
        }
        set_bnd3d_cpu(b, x);
    }
}

static void diffuse3d_cpu(int b, float *x, const float *x0, float diff_value, float dt_value) {
    if (diff_value == 0.0f) {
        memcpy(x, x0, volume_count() * sizeof(float));
        set_bnd3d_cpu(b, x);
        return;
    }
    float a = dt_value * diff_value * N * N;
    lin_solve3d_cpu(b, x, x0, a, 1.0f + 6.0f * a);
}

static void advect3d_cpu(int b, float *d, const float *d0,
                         const float *u_, const float *v_, const float *w_,
                         float dt_value) {
    float dt0 = dt_value * (float)N;
    for (int k = 1; k <= N; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= N; i++) {
                float x = (float)i - dt0 * u_[IX3(i, j, k)];
                float y = (float)j - dt0 * v_[IX3(i, j, k)];
                float z = (float)k - dt0 * w_[IX3(i, j, k)];
                if (x < 0.5f) x = 0.5f;
                if (x > N + 0.5f) x = N + 0.5f;
                if (y < 0.5f) y = 0.5f;
                if (y > N + 0.5f) y = N + 0.5f;
                if (z < 0.5f) z = 0.5f;
                if (z > N + 0.5f) z = N + 0.5f;

                int i0 = (int)x, j0 = (int)y, k0 = (int)z;
                int i1 = i0 + 1, j1 = j0 + 1, k1 = k0 + 1;
                float s1 = x - (float)i0, s0 = 1.0f - s1;
                float t1 = y - (float)j0, t0 = 1.0f - t1;
                float r1 = z - (float)k0, r0 = 1.0f - r1;

                d[IX3(i, j, k)] =
                    s0 * (t0 * (r0 * d0[IX3(i0, j0, k0)] + r1 * d0[IX3(i0, j0, k1)]) +
                          t1 * (r0 * d0[IX3(i0, j1, k0)] + r1 * d0[IX3(i0, j1, k1)])) +
                    s1 * (t0 * (r0 * d0[IX3(i1, j0, k0)] + r1 * d0[IX3(i1, j0, k1)]) +
                          t1 * (r0 * d0[IX3(i1, j1, k0)] + r1 * d0[IX3(i1, j1, k1)]));
            }
        }
    }
    set_bnd3d_cpu(b, d);
}

static void project3d_cpu(float *u_, float *v_, float *w_, float *p, float *div) {
    for (int k = 1; k <= N; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= N; i++) {
                div[IX3(i, j, k)] =
                    -0.5f * (u_[IX3(i + 1, j, k)] - u_[IX3(i - 1, j, k)] +
                             v_[IX3(i, j + 1, k)] - v_[IX3(i, j - 1, k)] +
                             w_[IX3(i, j, k + 1)] - w_[IX3(i, j, k - 1)]) / (float)N;
                p[IX3(i, j, k)] = 0.0f;
            }
        }
    }
    set_bnd3d_cpu(0, div);
    set_bnd3d_cpu(0, p);
    lin_solve3d_cpu(0, p, div, 1.0f, 6.0f);

    for (int k = 1; k <= N; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= N; i++) {
                u_[IX3(i, j, k)] -= 0.5f * (float)N * (p[IX3(i + 1, j, k)] - p[IX3(i - 1, j, k)]);
                v_[IX3(i, j, k)] -= 0.5f * (float)N * (p[IX3(i, j + 1, k)] - p[IX3(i, j - 1, k)]);
                w_[IX3(i, j, k)] -= 0.5f * (float)N * (p[IX3(i, j, k + 1)] - p[IX3(i, j, k - 1)]);
            }
        }
    }
    set_bnd3d_cpu(1, u_);
    set_bnd3d_cpu(2, v_);
    set_bnd3d_cpu(3, w_);
}

static void add_source3d_cpu(float *x, const float *s, float dt_value) {
    size_t count = volume_count();
    for (size_t i = 0; i < count; i++) {
        x[i] += dt_value * s[i];
    }
}

static void dens_step3d_cpu(float *x, float *x0, float *u_, float *v_, float *w_,
                            float diff_value, float dt_value) {
    add_source3d_cpu(x, x0, dt_value);
    SWAP(x0, x);
    diffuse3d_cpu(0, x, x0, diff_value, dt_value);
    SWAP(x0, x);
    advect3d_cpu(0, x, x0, u_, v_, w_, dt_value);
}

static void vel_step3d_cpu(float *u_, float *v_, float *w_,
                           float *u0, float *v0, float *w0,
                           float visc_value, float dt_value) {
    add_source3d_cpu(u_, u0, dt_value);
    add_source3d_cpu(v_, v0, dt_value);
    add_source3d_cpu(w_, w0, dt_value);
    SWAP(u0, u_);
    diffuse3d_cpu(1, u_, u0, visc_value, dt_value);
    SWAP(v0, v_);
    diffuse3d_cpu(2, v_, v0, visc_value, dt_value);
    SWAP(w0, w_);
    diffuse3d_cpu(3, w_, w0, visc_value, dt_value);
    project3d_cpu(u_, v_, w_, u0, v0);
    SWAP(u0, u_);
    SWAP(v0, v_);
    SWAP(w0, w_);
    advect3d_cpu(1, u_, u0, u0, v0, w0, dt_value);
    advect3d_cpu(2, v_, v0, u0, v0, w0, dt_value);
    advect3d_cpu(3, w_, w0, u0, v0, w0, dt_value);
    project3d_cpu(u_, v_, w_, u0, v0);
}

static void fade_fields3d_cpu(float *dens_, float *u_, float *v_, float *w_,
                              float density_damping, float velocity_damping) {
    size_t count = volume_count();
    for (size_t i = 0; i < count; i++) {
        dens_[i] *= density_damping;
        u_[i] *= velocity_damping;
        v_[i] *= velocity_damping;
        w_[i] *= velocity_damping;
    }
}

static void apply_sphere_obstacle3d_cpu(float *dens_, float *u_, float *v_, float *w_,
                                        float cx, float cy, float cz, float radius) {
    float r2 = radius * radius;
    float shell = radius + 3.0f;
    float shell2 = shell * shell;
    for (int k = 1; k <= N; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= N; i++) {
                float dx = (float)i - cx;
                float dy = (float)j - cy;
                float dz = (float)k - cz;
                float dist2 = dx * dx + dy * dy + dz * dz;
                int idx = IX3(i, j, k);
                if (dist2 <= r2) {
                    dens_[idx] = 0.0f;
                    u_[idx] = v_[idx] = w_[idx] = 0.0f;
                } else if (dist2 <= shell2 && dist2 > 1.0e-12f) {
                    float inv_dist = 1.0f / sqrtf(dist2);
                    float nx = dx * inv_dist;
                    float ny = dy * inv_dist;
                    float nz = dz * inv_dist;
                    float inward = u_[idx] * nx + v_[idx] * ny + w_[idx] * nz;
                    if (inward < 0.0f) {
                        u_[idx] -= inward * nx;
                        v_[idx] -= inward * ny;
                        w_[idx] -= inward * nz;
                    }
                }
            }
        }
    }
}

static void add_flow_source(float *dens_, float *u_, float *v_, float *w_, int frame) {
    float center[3], dir[3], tangent[3], bitangent[3];
    wind_source_frame(center, dir, tangent, bitangent);

    float half_depth = wind_source_depth_world * 0.5f;
    float radius2 = wind_source_radius_world * wind_source_radius_world;
    float extent = wind_source_radius_world + half_depth;
    int imin = world_to_cell_index(center[0] - extent);
    int imax = world_to_cell_index(center[0] + extent);
    int jmin = world_to_cell_index(center[1] - wind_source_radius_world);
    int jmax = world_to_cell_index(center[1] + wind_source_radius_world);
    int kmin = world_to_cell_index(center[2] - extent);
    int kmax = world_to_cell_index(center[2] + extent);
    float phase = (float)frame * 0.055f;

    for (int j = jmin; j <= jmax; j++) {
        float py = cell_to_world_coord(j);
        for (int k = kmin; k <= kmax; k++) {
            float pz = cell_to_world_coord(k);
            for (int i = imin; i <= imax; i++) {
                float px = cell_to_world_coord(i);
                float dx = px - center[0];
                float dy = py - center[1];
                float dz = pz - center[2];
                float along = dx * dir[0] + dy * dir[1] + dz * dir[2];
                if (fabsf(along) > half_depth) continue;

                float side = dx * tangent[0] + dy * tangent[1] + dz * tangent[2];
                float lift = dx * bitangent[0] + dy * bitangent[1] + dz * bitangent[2];
                float r2 = side * side + lift * lift;
                if (r2 > radius2) continue;

                float falloff = 1.0f - r2 / radius2;
                falloff *= falloff;
                falloff *= 1.0f - fabsf(along) / half_depth;
                float eddy = sinf(phase + (float)i * 0.075f) *
                             cosf(phase * 0.73f + (float)k * 0.061f);
                float swirl = 0.08f * auto_flow_speed * falloff;
                float eddy2 = 0.04f * sinf(phase * 1.21f + (float)j * 0.23f);
                int idx = IX3(i, j, k);
                dens_[idx] += source * 0.75f * falloff;
                u_[idx] += (dir[0] * auto_flow_speed +
                            tangent[0] * (swirl + 0.04f * eddy) +
                            bitangent[0] * eddy2) * falloff;
                v_[idx] += (dir[1] * auto_flow_speed +
                            tangent[1] * (swirl + 0.04f * eddy) +
                            bitangent[1] * eddy2) * falloff;
                w_[idx] += (dir[2] * auto_flow_speed +
                            tangent[2] * (swirl + 0.04f * eddy) +
                            bitangent[2] * eddy2) * falloff;
            }
        }
    }
}

static void set_benchmark_obstacle_pose(int frame) {
    float phase = ((float)frame + 1.0f) * 0.029f;
    obstacle_x = 0.22f * sinf(phase);
    obstacle_y = 0.10f * sinf(phase * 0.73f);
    obstacle_z = 0.18f * cosf(phase * 0.91f);
    clamp_obstacle_position();
}

static void compute_benchmark_metrics(double *density_sum,
                                      float *density_max,
                                      double *velocity_l2,
                                      double *divergence_l2,
                                      float *divergence_max) {
    double density_accum = 0.0;
    double velocity_accum = 0.0;
    double divergence_accum = 0.0;
    float max_density = 0.0f;
    float max_divergence = 0.0f;
    int count = 0;

    float obstacle_i = world_to_cell_coord(obstacle_x);
    float obstacle_j = world_to_cell_coord(obstacle_y);
    float obstacle_k = world_to_cell_coord(obstacle_z);
    float obstacle_r = obstacle_radius_cells();
    float obstacle_r2 = obstacle_r * obstacle_r;

    for (int k = 1; k <= N; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= N; i++) {
                float dx = (float)i - obstacle_i;
                float dy = (float)j - obstacle_j;
                float dz = (float)k - obstacle_k;
                if (dx * dx + dy * dy + dz * dz <= obstacle_r2) continue;

                int idx = IX3(i, j, k);
                float dens_value = dens[idx];
                float speed2 = u[idx] * u[idx] + v[idx] * v[idx] + w[idx] * w[idx];
                float div = 0.5f * (float)N *
                            (u[IX3(i + 1, j, k)] - u[IX3(i - 1, j, k)] +
                             v[IX3(i, j + 1, k)] - v[IX3(i, j - 1, k)] +
                             w[IX3(i, j, k + 1)] - w[IX3(i, j, k - 1)]);
                float abs_div = fabsf(div);

                density_accum += (double)dens_value;
                velocity_accum += (double)speed2;
                divergence_accum += (double)div * (double)div;
                if (dens_value > max_density) max_density = dens_value;
                if (abs_div > max_divergence) max_divergence = abs_div;
                count++;
            }
        }
    }

    if (count < 1) count = 1;
    *density_sum = density_accum;
    *density_max = max_density;
    *velocity_l2 = sqrt(velocity_accum / (double)count);
    *divergence_l2 = sqrt(divergence_accum / (double)count);
    *divergence_max = max_divergence;
}

static void make_benchmark_timestamp(char *buffer, size_t size) {
    time_t raw_time = time(NULL);
    struct tm local_time;
#ifdef _WIN32
    localtime_s(&local_time, &raw_time);
#else
    localtime_r(&raw_time, &local_time);
#endif
    snprintf(buffer, size, "%04d%02d%02d_%02d%02d%02d",
             local_time.tm_year + 1900, local_time.tm_mon + 1,
             local_time.tm_mday, local_time.tm_hour,
             local_time.tm_min, local_time.tm_sec);
}

static void write_benchmark_csv(const char *mode, const char *input_id,
                                const char *tag,
                                int grid_size, int warmup_frames, int measured_frames,
                                double source_ms, double step_ms, double total_ms,
                                double fps, double density_sum, float density_max,
                                double velocity_l2, double divergence_l2,
                                float divergence_max) {
    char timestamp[32];
    const char *filename = (strcmp(tag, "scaling") == 0) ?
        "benchmark_scaling_3d.csv" : "benchmark_3d.csv";
    make_benchmark_timestamp(timestamp, sizeof(timestamp));
    long long cells = (long long)grid_size * (long long)grid_size * (long long)grid_size;
    double scalar_field_mb = ((double)(grid_size + 2) * (double)(grid_size + 2) *
                              (double)(grid_size + 2) * (double)sizeof(float)) /
                             (1024.0 * 1024.0);
    double mcells_per_sec = (total_ms > 0.0) ?
        ((double)cells / (total_ms / 1000.0)) / 1000000.0 : 0.0;
    double ns_per_cell = (cells > 0) ? (total_ms * 1000000.0 / (double)cells) : 0.0;

#ifdef _WIN32
    wchar_t module_path[MAX_PATH];
    wchar_t exe_dir[MAX_PATH];
    wchar_t root_dir[MAX_PATH];
    wchar_t folder[MAX_PATH];
    wchar_t wide_filename[160];
    wchar_t csv_path[MAX_PATH];
    DWORD len = GetModuleFileNameW(NULL, module_path, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) return;
    wcscpy_s(exe_dir, MAX_PATH, module_path);
    wchar_t *last_slash = wcsrchr(exe_dir, L'\\');
    if (!last_slash) last_slash = wcsrchr(exe_dir, L'/');
    if (last_slash) *last_slash = L'\0';
    wcscpy_s(root_dir, MAX_PATH, exe_dir);
    last_slash = wcsrchr(root_dir, L'\\');
    if (!last_slash) last_slash = wcsrchr(root_dir, L'/');
    if (last_slash) *last_slash = L'\0';
    swprintf(folder, MAX_PATH, L"%ls\\benchmark_results", root_dir);
    if (_wmkdir(folder) != 0 && errno != EEXIST) return;
    MultiByteToWideChar(CP_UTF8, 0, filename, -1, wide_filename, 160);
    swprintf(csv_path, MAX_PATH, L"%ls\\%ls", folder, wide_filename);
    int need_header = 1;
    FILE *probe = _wfopen(csv_path, L"rb");
    if (probe) {
        fseek(probe, 0, SEEK_END);
        need_header = (ftell(probe) == 0);
        fclose(probe);
    }
    FILE *fp = _wfopen(csv_path, L"ab");
#else
    const char *folder = "../benchmark_results";
    if (mkdir(folder, 0777) != 0 && errno != EEXIST) return;
    char csv_path[512];
    snprintf(csv_path, sizeof(csv_path), "%s/%s", folder, filename);
    int need_header = 1;
    FILE *probe = fopen(csv_path, "rb");
    if (probe) {
        fseek(probe, 0, SEEK_END);
        need_header = (ftell(probe) == 0);
        fclose(probe);
    }
    FILE *fp = fopen(csv_path, "ab");
#endif
    if (!fp) return;
    if (need_header) {
        fprintf(fp, "timestamp,task,mode,input_id,N,dimension,cells,scalar_field_mb,warmup,frames,source_ms,step_ms,total_ms,fps,mcells_per_sec,ns_per_cell,density_sum,density_max,velocity_l2,divergence_l2,divergence_max\n");
    }
    fprintf(fp, "%s,3d,%s,%s,%d,3,%lld,%.6f,%d,%d,%.6f,%.6f,%.6f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6e,%.6e\n",
            timestamp, mode, input_id, grid_size, cells, scalar_field_mb,
            warmup_frames, measured_frames, source_ms, step_ms, total_ms, fps,
            mcells_per_sec, ns_per_cell, density_sum, density_max,
            velocity_l2, divergence_l2, divergence_max);
    fclose(fp);
    printf("benchmark_csv,benchmark_results/%s\n", filename);
}

static void init_benchmark_config(BenchmarkConfig *cfg) {
    cfg->enabled = 0;
    cfg->frames = 120;
    cfg->warmup = 20;
    cfg->grid_size = 32;
    strcpy(cfg->tag, "default");
    cfg->save_csv = 1;
}

static int parse_int_value_arg(const char *arg, const char *prefix, int *out) {
    size_t len = strlen(prefix);
    if (strncmp(arg, prefix, len) == 0 && arg[len] == '=') {
        *out = atoi(arg + len + 1);
        return 1;
    }
    return 0;
}

static void parse_benchmark_args(int argc, char **argv, BenchmarkConfig *cfg) {
    init_benchmark_config(cfg);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0 || strcmp(argv[i], "-b") == 0) {
            cfg->enabled = 1;
        } else if (strcmp(argv[i], "--bench-no-csv") == 0) {
            cfg->save_csv = 0;
            cfg->enabled = 1;
        } else if (strcmp(argv[i], "--bench-frames") == 0 && i + 1 < argc) {
            cfg->frames = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench-warmup") == 0 && i + 1 < argc) {
            cfg->warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench-size") == 0 && i + 1 < argc) {
            cfg->grid_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench-tag") == 0 && i + 1 < argc) {
            snprintf(cfg->tag, sizeof(cfg->tag), "%s", argv[++i]);
            cfg->enabled = 1;
        } else if (parse_int_value_arg(argv[i], "--bench-frames", &cfg->frames) ||
                   parse_int_value_arg(argv[i], "--bench-warmup", &cfg->warmup) ||
                   parse_int_value_arg(argv[i], "--bench-size", &cfg->grid_size)) {
            cfg->enabled = 1;
        }
    }
    if (cfg->frames < 1) cfg->frames = 1;
    if (cfg->warmup < 0) cfg->warmup = 0;
    if (cfg->grid_size < 8) cfg->grid_size = 8;
    if (cfg->grid_size > 256) cfg->grid_size = 256;
}

static int run_benchmark_cpu3d(const BenchmarkConfig *cfg) {
    N = cfg->grid_size;
    printf("=== Stable Fluids 3D (CPU benchmark) ===\n");
    printf("Benchmark input: deterministic_3d_wind_tunnel_v1, no GLUT, no user input\n");
    printf("Grid: N=%d, warmup=%d, frames=%d, %.1f MB per scalar field\n",
           N, cfg->warmup, cfg->frames,
           (double)(volume_count() * sizeof(float)) / (1024.0 * 1024.0));
    if (!allocate_data()) return 1;
    clear_data();

    int total_frames = cfg->warmup + cfg->frames;
    double source_seconds = 0.0;
    double step_seconds = 0.0;
    for (int frame = 0; frame < total_frames; frame++) {
        set_benchmark_obstacle_pose(frame);

        double source_t0 = now_seconds();
        clear_host_sources();
        add_flow_source(dens_prev, u_prev, v_prev, w_prev, frame);
        double source_t1 = now_seconds();

        double step_t0 = now_seconds();
        float obstacle_i = world_to_cell_coord(obstacle_x);
        float obstacle_j = world_to_cell_coord(obstacle_y);
        float obstacle_k = world_to_cell_coord(obstacle_z);
        float obstacle_r = obstacle_radius_cells();
        vel_step3d_cpu(u, v, w, u_prev, v_prev, w_prev, visc, dt);
        apply_sphere_obstacle3d_cpu(dens, u, v, w, obstacle_i, obstacle_j, obstacle_k, obstacle_r);
        dens_step3d_cpu(dens, dens_prev, u, v, w, diff, dt);
        apply_sphere_obstacle3d_cpu(dens, u, v, w, obstacle_i, obstacle_j, obstacle_k, obstacle_r);
        fade_fields3d_cpu(dens, u, v, w, dissipation, 0.992f);
        apply_sphere_obstacle3d_cpu(dens, u, v, w, obstacle_i, obstacle_j, obstacle_k, obstacle_r);
        double step_t1 = now_seconds();

        if (frame >= cfg->warmup) {
            source_seconds += source_t1 - source_t0;
            step_seconds += step_t1 - step_t0;
        }
    }

    double density_sum, velocity_l2, divergence_l2;
    float density_max, divergence_max;
    compute_benchmark_metrics(&density_sum, &density_max,
                              &velocity_l2, &divergence_l2, &divergence_max);

    double source_ms = source_seconds * 1000.0 / (double)cfg->frames;
    double step_ms = step_seconds * 1000.0 / (double)cfg->frames;
    double total_ms = source_ms + step_ms;
    double fps = (total_ms > 0.0) ? 1000.0 / total_ms : 0.0;

    printf("benchmark_header,mode,N,warmup,frames,source_ms,step_ms,total_ms,fps,density_sum,density_max,velocity_l2,divergence_l2,divergence_max\n");
    printf("benchmark_result,cpu3d,%d,%d,%d,%.6f,%.6f,%.6f,%.3f,%.6f,%.6f,%.6f,%.6e,%.6e\n",
           N, cfg->warmup, cfg->frames, source_ms, step_ms, total_ms, fps,
           density_sum, density_max, velocity_l2, divergence_l2, divergence_max);
    if (cfg->save_csv) {
        write_benchmark_csv("cpu3d", "deterministic_3d_wind_tunnel_v1",
                            cfg->tag, N, cfg->warmup, cfg->frames,
                            source_ms, step_ms, total_ms, fps,
                            density_sum, density_max, velocity_l2,
                            divergence_l2, divergence_max);
    }

    free_data();
    return 0;
}

int main(int argc, char **argv) {
    BenchmarkConfig cfg;
    parse_benchmark_args(argc, argv, &cfg);
    if (!cfg.enabled) {
        printf("CPU 3D is benchmark-only. Use --benchmark.\n");
        return 0;
    }
    return run_benchmark_cpu3d(&cfg);
}
