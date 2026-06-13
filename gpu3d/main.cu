// 3D Stable Fluids - GPU wind tunnel viewer

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <wchar.h>
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <time.h>
#include <sys/stat.h>
#endif

#ifdef __APPLE__
  #include <GLUT/glut.h>
#else
  #include <GL/glut.h>
#endif

#include "solver3d.h"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA ERROR %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err__));          \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

static int N = 64;
static float dt = 0.08f;
static float diff = 0.0f;
static float visc = 0.0f;
static float dissipation = 0.997f;
static float source = 420.0f;
static float color_speed_scale = 0.55f;
static float domain_half_extent = 1.45f;
static float auto_flow_speed = 1.15f;
static float obstacle_radius_world = 0.28f;
static float obstacle_x = 0.0f, obstacle_y = 0.0f, obstacle_z = 0.0f;
static float obstacle_move_speed = 0.045f;
static float wind_source_radius_world = 0.38f;
static float wind_source_depth_world = 0.30f;
static int control_mode = 0;  /* 0: camera, 1: obstacle */
static const float camera_fov_degrees = 58.0f;
static const float camera_near_clip = 0.01f;
static const float camera_far_clip = 100.0f;
static const float pi_f = 3.14159265f;

static float *h_dens_prev;
static float *h_u_prev, *h_v_prev, *h_w_prev;
static float *d_dens, *d_dens_prev;
static float *d_u, *d_v, *d_w, *d_u_prev, *d_v_prev, *d_w_prev;
static unsigned char *h_volume_pixels;
static uchar4 *d_volume_pixels;
static GLuint volume_tex = 0;
static int volume_tex_w = 0;
static int volume_tex_h = 0;

static int win_x = 900, win_y = 760;
static int mx, my;
static int paused = 0;
static int show_volume = 1;
static int auto_source = 1;
static int mode_toggle_down = 0;

static float cam_x = 0.0f, cam_y = 0.0f, cam_z = 4.2f;
static float cam_yaw = 0.0f, cam_pitch = 0.0f;
static float cam_speed = 0.08f;
static int right_look = 0;
static int keys[256];

static int frame_count = 0;
static double accum_time = 0.0;
static double last_fps_time = 0.0;
static double current_ms = 0.0;
static double current_fps = 0.0;

typedef struct BenchmarkConfig {
    int enabled;
    int frames;
    int warmup;
    int grid_size;
    char tag[32];
    int save_csv;
} BenchmarkConfig;

#define IX3(i, j, k) ((i) + (N + 2) * ((j) + (N + 2) * (k)))

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

static void toggle_control_mode(void) {
    control_mode = 1 - control_mode;
    printf("Control mode: %s\n", control_mode == 0 ? "camera" : "object");
}

static void poll_mode_toggle_key(void) {
#ifdef _WIN32
    int down = ((GetAsyncKeyState('M') & 0x8000) != 0) ||
               ((GetAsyncKeyState(VK_TAB) & 0x8000) != 0);
    if (down && !mode_toggle_down) {
        toggle_control_mode();
    }
    mode_toggle_down = down;
#endif
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

static void camera_basis(float *forward, float *right, float *up) {
    float cp = cosf(cam_pitch);
    forward[0] = sinf(cam_yaw) * cp;
    forward[1] = -sinf(cam_pitch);
    forward[2] = -cosf(cam_yaw) * cp;

    right[0] = cosf(cam_yaw);
    right[1] = 0.0f;
    right[2] = sinf(cam_yaw);

    up[0] = right[1] * forward[2] - right[2] * forward[1];
    up[1] = right[2] * forward[0] - right[0] * forward[2];
    up[2] = right[0] * forward[1] - right[1] * forward[0];
}

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

__device__ static float clamp01_device(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__device__ static float3 add3_device(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ static float3 sub3_device(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ static float3 mul3_device(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ static float dot3_device(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ static float3 normalize3_device(float3 v) {
    float len2 = dot3_device(v, v);
    if (len2 <= 1.0e-12f) return make_float3(0.0f, 0.0f, -1.0f);
    float inv_len = rsqrtf(len2);
    return mul3_device(v, inv_len);
}

__device__ static int intersect_box_device(float3 origin, float3 dir,
                                          float half_extent,
                                          float *t_near, float *t_far) {
    float t0 = -1.0e20f;
    float t1 =  1.0e20f;
    float o[3] = {origin.x, origin.y, origin.z};
    float d[3] = {dir.x, dir.y, dir.z};

    for (int axis = 0; axis < 3; axis++) {
        if (fabsf(d[axis]) < 1.0e-7f) {
            if (o[axis] < -half_extent || o[axis] > half_extent) return 0;
            continue;
        }
        float inv_d = 1.0f / d[axis];
        float a = (-half_extent - o[axis]) * inv_d;
        float b = ( half_extent - o[axis]) * inv_d;
        if (a > b) {
            float tmp = a;
            a = b;
            b = tmp;
        }
        if (a > t0) t0 = a;
        if (b < t1) t1 = b;
        if (t0 > t1) return 0;
    }

    *t_near = t0;
    *t_far = t1;
    return t1 > 0.0f;
}

__device__ static int intersect_sphere_device(float3 origin, float3 dir,
                                             float3 center, float radius,
                                             float *t_hit) {
    float3 oc = sub3_device(origin, center);
    float b = dot3_device(oc, dir);
    float c = dot3_device(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) return 0;

    float s = sqrtf(disc);
    float t = -b - s;
    if (t < 0.0f) t = -b + s;
    if (t < 0.0f) return 0;

    *t_hit = t;
    return 1;
}

__device__ static float sample_field_trilinear_device(int N,
                                                      const float *__restrict__ field,
                                                      float3 p,
                                                      float half_extent) {
    float domain = half_extent * 2.0f;
    float gx = ((p.x + half_extent) / domain) * (float)N + 0.5f;
    float gy = ((p.y + half_extent) / domain) * (float)N + 0.5f;
    float gz = ((p.z + half_extent) / domain) * (float)N + 0.5f;

    gx = fminf(fmaxf(gx, 1.0f), (float)N);
    gy = fminf(fmaxf(gy, 1.0f), (float)N);
    gz = fminf(fmaxf(gz, 1.0f), (float)N);

    int x0 = (int)floorf(gx);
    int y0 = (int)floorf(gy);
    int z0 = (int)floorf(gz);
    if (x0 >= N) x0 = N - 1;
    if (y0 >= N) y0 = N - 1;
    if (z0 >= N) z0 = N - 1;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    float tx = gx - (float)x0;
    float ty = gy - (float)y0;
    float tz = gz - (float)z0;

    float c000 = field[IX3(x0, y0, z0)];
    float c100 = field[IX3(x1, y0, z0)];
    float c010 = field[IX3(x0, y1, z0)];
    float c110 = field[IX3(x1, y1, z0)];
    float c001 = field[IX3(x0, y0, z1)];
    float c101 = field[IX3(x1, y0, z1)];
    float c011 = field[IX3(x0, y1, z1)];
    float c111 = field[IX3(x1, y1, z1)];

    float c00 = c000 + (c100 - c000) * tx;
    float c10 = c010 + (c110 - c010) * tx;
    float c01 = c001 + (c101 - c001) * tx;
    float c11 = c011 + (c111 - c011) * tx;
    float c0 = c00 + (c10 - c00) * ty;
    float c1 = c01 + (c11 - c01) * ty;
    return c0 + (c1 - c0) * tz;
}

__device__ static void hsv_to_rgb_device(float h, float s, float v,
                                         float *r, float *g, float *b) {
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    float rp = 0.0f, gp = 0.0f, bp = 0.0f;
    int band = (int)(h * 6.0f);
    switch (band) {
        case 0: rp = c; gp = x; break;
        case 1: rp = x; gp = c; break;
        case 2: gp = c; bp = x; break;
        case 3: gp = x; bp = c; break;
        case 4: rp = x; bp = c; break;
        default: rp = c; bp = x; break;
    }
    *r = rp + m;
    *g = gp + m;
    *b = bp + m;
}

__global__ void raymarch_volume_kernel(int N,
                                       const float *__restrict__ dens,
                                       const float *__restrict__ u,
                                       const float *__restrict__ v,
                                       const float *__restrict__ w,
                                       uchar4 *__restrict__ pixels,
                                       int width, int height,
                                       float3 cam_pos,
                                       float3 forward,
                                       float3 right,
                                       float3 up,
                                       float tan_half_fov,
                                       float aspect,
                                       float half_extent,
                                       float color_speed_scale,
                                       float3 obstacle_center,
                                       float obstacle_radius) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    float sx = (((float)px + 0.5f) / (float)width) * 2.0f - 1.0f;
    float sy = (((float)py + 0.5f) / (float)height) * 2.0f - 1.0f;
    float3 ray = add3_device(forward,
                             add3_device(mul3_device(right, sx * tan_half_fov * aspect),
                                         mul3_device(up, sy * tan_half_fov)));
    ray = normalize3_device(ray);

    float t0, t1;
    int out_idx = py * width + px;
    if (!intersect_box_device(cam_pos, ray, half_extent, &t0, &t1)) {
        pixels[out_idx] = make_uchar4(0, 0, 0, 0);
        return;
    }
    if (t0 < 0.0f) t0 = 0.0f;

    float t_obstacle;
    if (intersect_sphere_device(cam_pos, ray, obstacle_center,
                                obstacle_radius, &t_obstacle) &&
        t_obstacle > t0 && t_obstacle < t1) {
        t1 = t_obstacle;
    }

    float cell = (half_extent * 2.0f) / (float)N;
    float step = cell * 0.58f;
    float alpha = 0.0f;
    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    for (float t = t0; t <= t1 && alpha < 0.985f; t += step) {
        float3 p = add3_device(cam_pos, mul3_device(ray, t));
        float density = sample_field_trilinear_device(N, dens, p, half_extent);
        if (density <= 0.001f) continue;

        float vx = sample_field_trilinear_device(N, u, p, half_extent);
        float vy = sample_field_trilinear_device(N, v, p, half_extent);
        float vz = sample_field_trilinear_device(N, w, p, half_extent);
        float speed = sqrtf(vx * vx + vy * vy + vz * vz);
        float vel_t = clamp01_device(1.0f - expf(-speed * color_speed_scale));

        float tint_r, tint_g, tint_b;
        hsv_to_rgb_device(0.58f - 0.48f * vel_t, 0.78f, 1.0f,
                          &tint_r, &tint_g, &tint_b);
        float tint = 0.10f + 0.42f * vel_t;
        float brightness = clamp01_device(0.62f + 0.30f * vel_t);
        float sr = brightness * ((1.0f - tint) + tint * tint_r);
        float sg = brightness * ((1.0f - tint) + tint * tint_g);
        float sb = brightness * ((1.0f - tint) + tint * tint_b);

        float sample_alpha = 1.0f - expf(-density * 0.0075f * (step / cell));
        sample_alpha = fminf(sample_alpha, 0.14f);
        float trans = (1.0f - alpha) * sample_alpha;
        color.x += trans * sr;
        color.y += trans * sg;
        color.z += trans * sb;
        alpha += trans;
    }

    float inv_alpha = (alpha > 0.0001f) ? (1.0f / alpha) : 0.0f;
    unsigned char r = (unsigned char)(clamp01_device(color.x * inv_alpha) * 255.0f + 0.5f);
    unsigned char g = (unsigned char)(clamp01_device(color.y * inv_alpha) * 255.0f + 0.5f);
    unsigned char b = (unsigned char)(clamp01_device(color.z * inv_alpha) * 255.0f + 0.5f);
    unsigned char a = (unsigned char)(clamp01_device(alpha) * 255.0f + 0.5f);
    pixels[out_idx] = make_uchar4(r, g, b, a);
}

static void clear_host_sources(void) {
    size_t bytes = volume_count() * sizeof(float);
    memset(h_dens_prev, 0, bytes);
    memset(h_u_prev, 0, bytes);
    memset(h_v_prev, 0, bytes);
    memset(h_w_prev, 0, bytes);
}

static void clear_data(void) {
    size_t bytes = volume_count() * sizeof(float);
    clear_host_sources();
    CUDA_CHECK(cudaMemset(d_dens, 0, bytes));
    CUDA_CHECK(cudaMemset(d_u, 0, bytes));
    CUDA_CHECK(cudaMemset(d_v, 0, bytes));
    CUDA_CHECK(cudaMemset(d_w, 0, bytes));
    CUDA_CHECK(cudaMemset(d_dens_prev, 0, bytes));
    CUDA_CHECK(cudaMemset(d_u_prev, 0, bytes));
    CUDA_CHECK(cudaMemset(d_v_prev, 0, bytes));
    CUDA_CHECK(cudaMemset(d_w_prev, 0, bytes));
}

static int allocate_data(void) {
    size_t bytes = volume_count() * sizeof(float);
    h_dens_prev = (float *)malloc(bytes);
    h_u_prev = (float *)malloc(bytes);
    h_v_prev = (float *)malloc(bytes);
    h_w_prev = (float *)malloc(bytes);
    if (!h_dens_prev || !h_u_prev || !h_v_prev || !h_w_prev) {
        fprintf(stderr, "ERROR: out of host memory\n");
        return 0;
    }

    CUDA_CHECK(cudaMalloc(&d_dens, bytes));
    CUDA_CHECK(cudaMalloc(&d_dens_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_w, bytes));
    CUDA_CHECK(cudaMalloc(&d_u_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_v_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_w_prev, bytes));
    init_solver3d(N);
    return 1;
}

static void release_volume_render_target(void) {
    free(h_volume_pixels);
    h_volume_pixels = NULL;
    if (d_volume_pixels) {
        CUDA_CHECK(cudaFree(d_volume_pixels));
        d_volume_pixels = NULL;
    }
    if (volume_tex) {
        glDeleteTextures(1, &volume_tex);
        volume_tex = 0;
    }
    volume_tex_w = 0;
    volume_tex_h = 0;
}

static void ensure_volume_render_target(void) {
    int w_px = (win_x > 1) ? win_x : 1;
    int h_px = (win_y > 1) ? win_y : 1;
    if (volume_tex && w_px == volume_tex_w && h_px == volume_tex_h) return;

    release_volume_render_target();
    size_t pixel_bytes = (size_t)w_px * h_px * 4;
    h_volume_pixels = (unsigned char *)malloc(pixel_bytes);
    if (!h_volume_pixels) {
        fprintf(stderr, "ERROR: out of host memory for volume render target\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMalloc(&d_volume_pixels, (size_t)w_px * h_px * sizeof(uchar4)));

    glGenTextures(1, &volume_tex);
    glBindTexture(GL_TEXTURE_2D, volume_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w_px, h_px, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    volume_tex_w = w_px;
    volume_tex_h = h_px;
}

static void free_data(void) {
    release_volume_render_target();
    free(h_dens_prev);
    free(h_u_prev); free(h_v_prev); free(h_w_prev);
    CUDA_CHECK(cudaFree(d_dens));
    CUDA_CHECK(cudaFree(d_dens_prev));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_u_prev));
    CUDA_CHECK(cudaFree(d_v_prev));
    CUDA_CHECK(cudaFree(d_w_prev));
    free_solver3d();
}

static void add_flow_source(float *dens, float *u, float *v, float *w) {
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
    float phase = (float)frame_count * 0.055f;

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
                dens[idx] += source * 0.75f * falloff;
                u[idx] += (dir[0] * auto_flow_speed +
                           tangent[0] * (swirl + 0.04f * eddy) +
                           bitangent[0] * eddy2) * falloff;
                v[idx] += (dir[1] * auto_flow_speed +
                           tangent[1] * (swirl + 0.04f * eddy) +
                           bitangent[1] * eddy2) * falloff;
                w[idx] += (dir[2] * auto_flow_speed +
                           tangent[2] * (swirl + 0.04f * eddy) +
                           bitangent[2] * eddy2) * falloff;
            }
        }
    }
}

static void get_from_ui(void) {
    clear_host_sources();
    if (auto_source) {
        add_flow_source(h_dens_prev, h_u_prev, h_v_prev, h_w_prev);
    }
}

static void update_controls(void) {
    float cy = cosf(cam_yaw), sy = sinf(cam_yaw);
    float fx = sy, fz = -cy;
    float rx = cy, rz = sy;
    if (control_mode == 0) {
        if (keys['w'] || keys['W']) { cam_x += fx * cam_speed; cam_z += fz * cam_speed; }
        if (keys['s'] || keys['S']) { cam_x -= fx * cam_speed; cam_z -= fz * cam_speed; }
        if (keys['d'] || keys['D']) { cam_x += rx * cam_speed; cam_z += rz * cam_speed; }
        if (keys['a'] || keys['A']) { cam_x -= rx * cam_speed; cam_z -= rz * cam_speed; }
        if (keys['e'] || keys['E']) cam_y += cam_speed;
        if (keys['q'] || keys['Q']) cam_y -= cam_speed;
    } else {
        if (keys['w'] || keys['W']) { obstacle_x += fx * obstacle_move_speed; obstacle_z += fz * obstacle_move_speed; }
        if (keys['s'] || keys['S']) { obstacle_x -= fx * obstacle_move_speed; obstacle_z -= fz * obstacle_move_speed; }
        if (keys['d'] || keys['D']) { obstacle_x += rx * obstacle_move_speed; obstacle_z += rz * obstacle_move_speed; }
        if (keys['a'] || keys['A']) { obstacle_x -= rx * obstacle_move_speed; obstacle_z -= rz * obstacle_move_speed; }
        if (keys['e'] || keys['E']) obstacle_y += obstacle_move_speed;
        if (keys['q'] || keys['Q']) obstacle_y -= obstacle_move_speed;
        clamp_obstacle_position();
    }
}

static void draw_text(float x, float y, const char *str) {
    glColor3f(1.0f, 1.0f, 0.2f);
    glRasterPos2f(x, y);
    for (const char *c = str; *c; c++) {
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *c);
    }
}

static void apply_camera_3d(void) {
    float aspect = (win_y > 0) ? (float)win_x / (float)win_y : 1.0f;
    float forward[3], right[3], up[3];
    camera_basis(forward, right, up);

    glViewport(0, 0, win_x > 0 ? win_x : 1, win_y > 0 ? win_y : 1);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(camera_fov_degrees, aspect, camera_near_clip, camera_far_clip);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(cam_x, cam_y, cam_z,
              cam_x + forward[0], cam_y + forward[1], cam_z + forward[2],
              up[0], up[1], up[2]);
}

static void draw_raymarched_volume(void) {
    if (!show_volume) return;
    ensure_volume_render_target();

    float forward_arr[3], right_arr[3], up_arr[3];
    camera_basis(forward_arr, right_arr, up_arr);
    float3 cam_pos = make_float3(cam_x, cam_y, cam_z);
    float3 forward = make_float3(forward_arr[0], forward_arr[1], forward_arr[2]);
    float3 right = make_float3(right_arr[0], right_arr[1], right_arr[2]);
    float3 up = make_float3(up_arr[0], up_arr[1], up_arr[2]);
    float3 obstacle_center = make_float3(obstacle_x, obstacle_y, obstacle_z);
    float aspect = (volume_tex_h > 0) ? (float)volume_tex_w / (float)volume_tex_h : 1.0f;
    float fov = camera_fov_degrees * 3.14159265f / 180.0f;

    dim3 threads(16, 16);
    dim3 blocks((volume_tex_w + 15) / 16, (volume_tex_h + 15) / 16);
    raymarch_volume_kernel<<<blocks, threads>>>(N, d_dens, d_u, d_v, d_w,
                                                d_volume_pixels,
                                                volume_tex_w, volume_tex_h,
                                                cam_pos, forward, right, up,
                                                tanf(fov * 0.5f), aspect,
                                                domain_half_extent,
                                                color_speed_scale,
                                                obstacle_center,
                                                obstacle_radius_world);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaMemcpy(h_volume_pixels, d_volume_pixels,
                          (size_t)volume_tex_w * volume_tex_h * 4,
                          cudaMemcpyDeviceToHost));

    glBindTexture(GL_TEXTURE_2D, volume_tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, volume_tex_w, volume_tex_h,
                    GL_RGBA, GL_UNSIGNED_BYTE, h_volume_pixels);

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, volume_tex);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

    glViewport(0, 0, win_x > 0 ? win_x : 1, win_y > 0 ? win_y : 1);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 1.0f);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
}

static void draw_cube_bounds(void) {
    float h = domain_half_extent;
    glDisable(GL_BLEND);
    glLineWidth(1.5f);
    glColor3f(0.28f, 0.34f, 0.40f);

    glBegin(GL_LINE_LOOP);
    glVertex3f(-h, -h, -h); glVertex3f(h, -h, -h);
    glVertex3f(h, h, -h); glVertex3f(-h, h, -h);
    glEnd();
    glBegin(GL_LINE_LOOP);
    glVertex3f(-h, -h, h); glVertex3f(h, -h, h);
    glVertex3f(h, h, h); glVertex3f(-h, h, h);
    glEnd();
    glBegin(GL_LINES);
    glVertex3f(-h, -h, -h); glVertex3f(-h, -h, h);
    glVertex3f(h, -h, -h); glVertex3f(h, -h, h);
    glVertex3f(h, h, -h); glVertex3f(h, h, h);
    glVertex3f(-h, h, -h); glVertex3f(-h, h, h);
    glEnd();

    glLineWidth(2.0f);
    glBegin(GL_LINES);
    float axis_o = -h * 1.16f;
    float axis_l = h * 0.34f;
    glColor3f(1.0f, 0.25f, 0.20f);
    glVertex3f(axis_o, axis_o, axis_o); glVertex3f(axis_o + axis_l, axis_o, axis_o);
    glColor3f(0.35f, 1.0f, 0.35f);
    glVertex3f(axis_o, axis_o, axis_o); glVertex3f(axis_o, axis_o + axis_l, axis_o);
    glColor3f(0.35f, 0.55f, 1.0f);
    glVertex3f(axis_o, axis_o, axis_o); glVertex3f(axis_o, axis_o, axis_o + axis_l);
    glEnd();
    glLineWidth(1.0f);
}

static void draw_wind_source_marker(void) {
    float center[3], dir[3], tangent[3], bitangent[3];
    wind_source_frame(center, dir, tangent, bitangent);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glLineWidth(2.0f);
    glColor3f(0.10f, 0.85f, 1.0f);

    glBegin(GL_LINE_LOOP);
    for (int s = 0; s < 48; s++) {
        float a = 2.0f * pi_f * (float)s / 48.0f;
        float side = cosf(a) * wind_source_radius_world;
        float lift = sinf(a) * wind_source_radius_world;
        glVertex3f(center[0] + tangent[0] * side + bitangent[0] * lift,
                   center[1] + tangent[1] * side + bitangent[1] * lift,
                   center[2] + tangent[2] * side + bitangent[2] * lift);
    }
    glEnd();

    glBegin(GL_LINES);
    float tip[3] = {
        center[0] + dir[0] * 0.46f,
        center[1] + dir[1] * 0.46f,
        center[2] + dir[2] * 0.46f
    };
    float head_a[3] = {
        center[0] + dir[0] * 0.34f + tangent[0] * 0.08f + bitangent[0] * 0.05f,
        center[1] + dir[1] * 0.34f + tangent[1] * 0.08f + bitangent[1] * 0.05f,
        center[2] + dir[2] * 0.34f + tangent[2] * 0.08f + bitangent[2] * 0.05f
    };
    float head_b[3] = {
        center[0] + dir[0] * 0.34f - tangent[0] * 0.08f - bitangent[0] * 0.05f,
        center[1] + dir[1] * 0.34f - tangent[1] * 0.08f - bitangent[1] * 0.05f,
        center[2] + dir[2] * 0.34f - tangent[2] * 0.08f - bitangent[2] * 0.05f
    };
    glVertex3f(center[0], center[1], center[2]);
    glVertex3f(tip[0], tip[1], tip[2]);
    glVertex3f(tip[0], tip[1], tip[2]);
    glVertex3f(head_a[0], head_a[1], head_a[2]);
    glVertex3f(tip[0], tip[1], tip[2]);
    glVertex3f(head_b[0], head_b[1], head_b[2]);
    glEnd();
    glLineWidth(1.0f);
}

static void draw_fixed_obstacle(void) {
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glPushMatrix();
    glTranslatef(obstacle_x, obstacle_y, obstacle_z);
    glColor3f(0.12f, 0.14f, 0.16f);
    glutSolidSphere(obstacle_radius_world, 36, 20);
    glColor3f(0.55f, 0.62f, 0.70f);
    glutWireSphere(obstacle_radius_world * 1.01f, 24, 12);
    glPopMatrix();
}

static void display(void) {
    glViewport(0, 0, win_x > 0 ? win_x : 1, win_y > 0 ? win_y : 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_CULL_FACE);

    apply_camera_3d();
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    draw_fixed_obstacle();

    draw_raymarched_volume();

    apply_camera_3d();
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    draw_cube_bounds();
    draw_wind_source_marker();

    glDepthMask(GL_TRUE);
    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, win_x > 0 ? win_x : 1, win_y > 0 ? win_y : 1);
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    char line[256];
    snprintf(line, sizeof(line), "3D Wind Tunnel N=%d  [M/Tab] Mode=%s  fixed source  box=%.2f%s%s  %.2f ms  FPS=%.1f%s",
             N, control_mode == 0 ? "CAMERA" : "OBJECT",
             domain_size(),
             show_volume ? " smoke" : "", auto_source ? " flow" : "",
             current_ms, current_fps,
             paused ? "  [PAUSED]" : "");
    draw_text(0.01f, 0.97f, line);
    snprintf(line, sizeof(line), "WASD/QE move selected target  wheel speed  RMB look  b volume  o flow  p pause  c clear");
    draw_text(0.01f, 0.94f, line);

    glutSwapBuffers();
}

static void idle(void) {
    poll_mode_toggle_key();
    update_controls();
    if (!paused) {
        get_from_ui();
        size_t bytes = volume_count() * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_dens_prev, h_dens_prev, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u_prev, h_u_prev, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_prev, h_v_prev, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w_prev, h_w_prev, bytes, cudaMemcpyHostToDevice));

        double t0 = now_seconds();
        float obstacle_i = world_to_cell_coord(obstacle_x);
        float obstacle_j = world_to_cell_coord(obstacle_y);
        float obstacle_k = world_to_cell_coord(obstacle_z);
        float obstacle_r = obstacle_radius_cells();
        vel_step3d(N, d_u, d_v, d_w, d_u_prev, d_v_prev, d_w_prev, visc, dt);
        apply_sphere_obstacle3d(N, d_dens, d_u, d_v, d_w,
                                obstacle_i, obstacle_j, obstacle_k, obstacle_r);
        dens_step3d(N, d_dens, d_dens_prev, d_u, d_v, d_w, diff, dt);
        apply_sphere_obstacle3d(N, d_dens, d_u, d_v, d_w,
                                obstacle_i, obstacle_j, obstacle_k, obstacle_r);
        fade_fields3d(N, d_dens, d_u, d_v, d_w, dissipation, 0.992f);
        apply_sphere_obstacle3d(N, d_dens, d_u, d_v, d_w,
                                obstacle_i, obstacle_j, obstacle_k, obstacle_r);
        CUDA_CHECK(cudaDeviceSynchronize());
        double t1 = now_seconds();

        accum_time += t1 - t0;
        frame_count++;

        double now = now_seconds();
        if (now - last_fps_time > 0.5) {
            current_ms = (accum_time / frame_count) * 1000.0;
            current_fps = frame_count / (accum_time > 0.0 ? accum_time : 1.0);
            printf("3D wind tunnel N=%d GPU=%.3f ms fps=%.1f mode=%s obj=(%.2f %.2f %.2f)\n",
                   N, current_ms, current_fps,
                   control_mode == 0 ? "camera" : "object",
                   obstacle_x, obstacle_y, obstacle_z);
            accum_time = 0.0;
            frame_count = 0;
            last_fps_time = now;
        }
    }
    glutPostRedisplay();
}

static void reshape(int w, int h) {
    win_x = w;
    win_y = h;
    glViewport(0, 0, w, h);
}

static void mouse(int button, int state, int x, int y) {
    mx = x;
    my = y;
    if (button == GLUT_RIGHT_BUTTON) {
        right_look = (state == GLUT_DOWN);
    }
    if (button == 3 && state == GLUT_DOWN) cam_speed *= 1.15f;
    if (button == 4 && state == GLUT_DOWN) cam_speed /= 1.15f;
}

static void motion(int x, int y) {
    if (right_look) {
        cam_yaw += (x - mx) * 0.006f;
        cam_pitch += (y - my) * 0.006f;
        if (cam_pitch > 1.35f) cam_pitch = 1.35f;
        if (cam_pitch < -1.35f) cam_pitch = -1.35f;
    }
    mx = x;
    my = y;
}

static void keyboard(unsigned char key, int x, int y) {
    (void)x; (void)y;
    if (key == 'm' || key == 'M' || key == '\t') {
#ifndef _WIN32
        if (!mode_toggle_down) {
            toggle_control_mode();
        }
        mode_toggle_down = 1;
#endif
        return;
    }

    int was_down = keys[key];
    keys[key] = 1;
    switch (key) {
        case 27: free_data(); exit(0);
        case 'p': case 'P': if (!was_down) paused = !paused; break;
        case 'c': case 'C': if (!was_down) clear_data(); break;
        case 'b': case 'B': if (!was_down) show_volume = !show_volume; break;
        case 'o': case 'O': if (!was_down) auto_source = !auto_source; break;
        case 'f': case 'F':
            cam_x = 0.0f; cam_y = 0.0f; cam_z = 4.2f; cam_yaw = 0.0f; cam_pitch = 0.0f;
            break;
    }
}

static void keyboard_up(unsigned char key, int x, int y) {
    (void)x; (void)y;
    if (key == 'm' || key == 'M' || key == '\t') {
#ifndef _WIN32
        mode_toggle_down = 0;
#endif
        return;
    }
    keys[key] = 0;
}

static void prompt_grid_size(void) {
    char line[64];
    printf("3D grid size N [default %d, suggested 48/64/96]: ", N);
    fflush(stdout);
    if (!fgets(line, sizeof(line), stdin)) return;
    if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') return;
    int value = atoi(line);
    if (value < 16) value = 16;
    if (value > 128) value = 128;
    N = value;
}

static void init_benchmark_config(BenchmarkConfig *cfg) {
    cfg->enabled = 0;
    cfg->frames = 120;
    cfg->warmup = 20;
    cfg->grid_size = 64;
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
    if (cfg->grid_size < 16) cfg->grid_size = 16;
    if (cfg->grid_size > 256) cfg->grid_size = 256;
}

static void set_benchmark_obstacle_pose_gpu3d(int frame) {
    float phase = ((float)frame + 1.0f) * 0.029f;
    obstacle_x = 0.22f * sinf(phase);
    obstacle_y = 0.10f * sinf(phase * 0.73f);
    obstacle_z = 0.18f * cosf(phase * 0.91f);
    clamp_obstacle_position();
}

static void prepare_benchmark_sources_gpu3d(int frame) {
    frame_count = frame;
    clear_host_sources();
    add_flow_source(h_dens_prev, h_u_prev, h_v_prev, h_w_prev);

    size_t bytes = volume_count() * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_dens_prev, h_dens_prev, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_prev, h_u_prev, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_prev, h_v_prev, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w_prev, h_w_prev, bytes, cudaMemcpyHostToDevice));
}

static void compute_benchmark_metrics_gpu3d(double *density_sum,
                                            float *density_max,
                                            double *velocity_l2,
                                            double *divergence_l2,
                                            float *divergence_max) {
    size_t bytes = volume_count() * sizeof(float);
    CUDA_CHECK(cudaMemcpy(h_dens_prev, d_dens, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_u_prev, d_u, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v_prev, d_v, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w_prev, d_w, bytes, cudaMemcpyDeviceToHost));

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
                float dens_value = h_dens_prev[idx];
                float speed2 = h_u_prev[idx] * h_u_prev[idx] +
                               h_v_prev[idx] * h_v_prev[idx] +
                               h_w_prev[idx] * h_w_prev[idx];
                float div = 0.5f * (float)N *
                            (h_u_prev[IX3(i + 1, j, k)] - h_u_prev[IX3(i - 1, j, k)] +
                             h_v_prev[IX3(i, j + 1, k)] - h_v_prev[IX3(i, j - 1, k)] +
                             h_w_prev[IX3(i, j, k + 1)] - h_w_prev[IX3(i, j, k - 1)]);
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
             local_time.tm_year + 1900,
             local_time.tm_mon + 1,
             local_time.tm_mday,
             local_time.tm_hour,
             local_time.tm_min,
             local_time.tm_sec);
}

static void write_benchmark_csv_gpu3d(const char *mode,
                                      const char *input_id,
                                      const char *tag,
                                      int grid_size,
                                      int warmup_frames,
                                      int measured_frames,
                                      double source_ms,
                                      double step_ms,
                                      double total_ms,
                                      double fps,
                                      double density_sum,
                                      float density_max,
                                      double velocity_l2,
                                      double divergence_l2,
                                      float divergence_max,
                                      double source_add_ms,
                                      double diffuse_ms,
                                      double project_ms,
                                      double advect_ms,
                                      double boundary_ms,
                                      double obstacle_ms,
                                      double fade_ms,
                                      double other_ms) {
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
    if (len == 0 || len >= MAX_PATH) {
        fwprintf(stderr, L"WARNING: could not resolve executable path for benchmark CSV\n");
        return;
    }
    wcscpy_s(exe_dir, MAX_PATH, module_path);
    wchar_t *last_slash = wcsrchr(exe_dir, L'\\');
    if (!last_slash) last_slash = wcsrchr(exe_dir, L'/');
    if (last_slash) {
        *last_slash = L'\0';
    }
    wcscpy_s(root_dir, MAX_PATH, exe_dir);
    last_slash = wcsrchr(root_dir, L'\\');
    if (!last_slash) last_slash = wcsrchr(root_dir, L'/');
    if (last_slash) {
        *last_slash = L'\0';
    }

    swprintf(folder, MAX_PATH, L"%ls\\benchmark_results", root_dir);
    if (_wmkdir(folder) != 0 && errno != EEXIST) {
        fwprintf(stderr, L"WARNING: could not create benchmark folder: %ls\n", folder);
        return;
    }
    MultiByteToWideChar(CP_UTF8, 0, filename, -1, wide_filename, 160);
    swprintf(csv_path, MAX_PATH, L"%ls\\%ls", folder, wide_filename);
    int need_header = 1;
    FILE *probe = _wfopen(csv_path, L"rb");
    if (probe) {
        char first_line[1024] = {0};
        int has_header = (fgets(first_line, sizeof(first_line), probe) != NULL);
        fseek(probe, 0, SEEK_END);
        long file_size = ftell(probe);
        int stale_header = has_header && strstr(first_line, "source_add_ms") == NULL;
        need_header = (file_size == 0) || stale_header;
        fclose(probe);
        if (stale_header) {
            wchar_t wide_timestamp[32];
            wchar_t legacy_path[MAX_PATH];
            MultiByteToWideChar(CP_UTF8, 0, timestamp, -1, wide_timestamp, 32);
            swprintf(legacy_path, MAX_PATH, L"%ls\\%ls.legacy_%ls", folder, wide_filename, wide_timestamp);
            _wrename(csv_path, legacy_path);
        }
    }
    FILE *fp = _wfopen(csv_path, L"ab");
#else
    const char *folder = "../benchmark_results";
    if (mkdir(folder, 0777) != 0 && errno != EEXIST) {
        fprintf(stderr, "WARNING: could not create benchmark folder: %s\n", folder);
        return;
    }
    char csv_path[512];
    snprintf(csv_path, sizeof(csv_path), "%s/%s", folder, filename);
    int need_header = 1;
    FILE *probe = fopen(csv_path, "rb");
    if (probe) {
        char first_line[1024] = {0};
        int has_header = (fgets(first_line, sizeof(first_line), probe) != NULL);
        fseek(probe, 0, SEEK_END);
        long file_size = ftell(probe);
        int stale_header = has_header && strstr(first_line, "source_add_ms") == NULL;
        need_header = (file_size == 0) || stale_header;
        fclose(probe);
        if (stale_header) {
            char legacy_path[512];
            snprintf(legacy_path, sizeof(legacy_path), "%s/%s.legacy_%s", folder, filename, timestamp);
            rename(csv_path, legacy_path);
        }
    }
    FILE *fp = fopen(csv_path, "ab");
#endif
    if (!fp) {
        fprintf(stderr, "WARNING: could not write benchmark CSV: %s\n", filename);
        return;
    }

    if (need_header) {
        fprintf(fp, "timestamp,task,mode,input_id,N,dimension,cells,scalar_field_mb,warmup,frames,source_ms,step_ms,total_ms,fps,mcells_per_sec,ns_per_cell,density_sum,density_max,velocity_l2,divergence_l2,divergence_max,source_add_ms,diffuse_ms,project_ms,advect_ms,boundary_ms,obstacle_ms,fade_ms,other_ms\n");
    }
    fprintf(fp, "%s,3d,%s,%s,%d,3,%lld,%.6f,%d,%d,%.6f,%.6f,%.6f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6e,%.6e,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            timestamp, mode, input_id, grid_size, cells, scalar_field_mb,
            warmup_frames, measured_frames,
            source_ms, step_ms, total_ms, fps,
            mcells_per_sec, ns_per_cell,
            density_sum, density_max, velocity_l2,
            divergence_l2, divergence_max,
            source_add_ms, diffuse_ms, project_ms, advect_ms,
            boundary_ms, obstacle_ms, fade_ms, other_ms);
    fclose(fp);

    printf("benchmark_csv,benchmark_results/%s\n", filename);
}

static int run_benchmark_gpu3d(const BenchmarkConfig *cfg) {
    N = cfg->grid_size;
    printf("=== Stable Fluids 3D (GPU/CUDA benchmark) ===\n");
    printf("Benchmark input: deterministic_3d_wind_tunnel_v1, no GLUT, no user input\n");
    printf("Grid: N=%d, warmup=%d, frames=%d, %.1f MB per scalar field\n",
           N, cfg->warmup, cfg->frames,
           (double)(volume_count() * sizeof(float)) / (1024.0 * 1024.0));

    if (!allocate_data()) return 1;
    clear_data();

    int total_frames = cfg->warmup + cfg->frames;
    double source_seconds = 0.0;
    double step_seconds = 0.0;
    SolverProfile3D profile_sum = {0};

    for (int frame = 0; frame < total_frames; frame++) {
        set_benchmark_obstacle_pose_gpu3d(frame);

        double source_t0 = now_seconds();
        prepare_benchmark_sources_gpu3d(frame);
        CUDA_CHECK(cudaDeviceSynchronize());
        double source_t1 = now_seconds();

        SolverProfile3D frame_profile = {0};
        solver3d_set_profile(&frame_profile);
        double step_t0 = now_seconds();
        float obstacle_i = world_to_cell_coord(obstacle_x);
        float obstacle_j = world_to_cell_coord(obstacle_y);
        float obstacle_k = world_to_cell_coord(obstacle_z);
        float obstacle_r = obstacle_radius_cells();
        vel_step3d(N, d_u, d_v, d_w, d_u_prev, d_v_prev, d_w_prev, visc, dt);
        apply_sphere_obstacle3d(N, d_dens, d_u, d_v, d_w,
                                obstacle_i, obstacle_j, obstacle_k, obstacle_r);
        dens_step3d(N, d_dens, d_dens_prev, d_u, d_v, d_w, diff, dt);
        apply_sphere_obstacle3d(N, d_dens, d_u, d_v, d_w,
                                obstacle_i, obstacle_j, obstacle_k, obstacle_r);
        fade_fields3d(N, d_dens, d_u, d_v, d_w, dissipation, 0.992f);
        apply_sphere_obstacle3d(N, d_dens, d_u, d_v, d_w,
                                obstacle_i, obstacle_j, obstacle_k, obstacle_r);
        CUDA_CHECK(cudaDeviceSynchronize());
        double step_t1 = now_seconds();
        solver3d_set_profile(NULL);

        if (frame >= cfg->warmup) {
            source_seconds += source_t1 - source_t0;
            step_seconds += step_t1 - step_t0;
            profile_sum.source_add_ms += frame_profile.source_add_ms;
            profile_sum.diffuse_ms += frame_profile.diffuse_ms;
            profile_sum.project_ms += frame_profile.project_ms;
            profile_sum.advect_ms += frame_profile.advect_ms;
            profile_sum.boundary_ms += frame_profile.boundary_ms;
            profile_sum.obstacle_ms += frame_profile.obstacle_ms;
            profile_sum.fade_ms += frame_profile.fade_ms;
        }
    }

    double density_sum;
    double velocity_l2;
    double divergence_l2;
    float density_max;
    float divergence_max;
    compute_benchmark_metrics_gpu3d(&density_sum, &density_max,
                                    &velocity_l2, &divergence_l2,
                                    &divergence_max);

    double source_ms = source_seconds * 1000.0 / (double)cfg->frames;
    double step_ms = step_seconds * 1000.0 / (double)cfg->frames;
    double total_ms = source_ms + step_ms;
    double fps = (total_ms > 0.0) ? 1000.0 / total_ms : 0.0;
    double inv_frames = 1.0 / (double)cfg->frames;
    double source_add_ms = profile_sum.source_add_ms * inv_frames;
    double diffuse_ms = profile_sum.diffuse_ms * inv_frames;
    double project_ms = profile_sum.project_ms * inv_frames;
    double advect_ms = profile_sum.advect_ms * inv_frames;
    double boundary_ms = profile_sum.boundary_ms * inv_frames;
    double obstacle_ms = profile_sum.obstacle_ms * inv_frames;
    double fade_ms = profile_sum.fade_ms * inv_frames;
    double accounted_ms = source_add_ms + diffuse_ms + project_ms + advect_ms +
                          boundary_ms + obstacle_ms + fade_ms;
    double other_ms = step_ms - accounted_ms;

    printf("benchmark_header,mode,N,warmup,frames,source_ms,step_ms,total_ms,fps,density_sum,density_max,velocity_l2,divergence_l2,divergence_max,source_add_ms,diffuse_ms,project_ms,advect_ms,boundary_ms,obstacle_ms,fade_ms,other_ms\n");
    printf("benchmark_result,gpu3d,%d,%d,%d,%.6f,%.6f,%.6f,%.3f,%.6f,%.6f,%.6f,%.6e,%.6e,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
           N, cfg->warmup, cfg->frames,
           source_ms, step_ms, total_ms, fps,
           density_sum, density_max, velocity_l2,
           divergence_l2, divergence_max,
           source_add_ms, diffuse_ms, project_ms, advect_ms,
           boundary_ms, obstacle_ms, fade_ms, other_ms);
    if (cfg->save_csv) {
        write_benchmark_csv_gpu3d("gpu3d", "deterministic_3d_wind_tunnel_v1",
                                  cfg->tag, N, cfg->warmup, cfg->frames,
                                  source_ms, step_ms, total_ms, fps,
                                  density_sum, density_max, velocity_l2,
                                  divergence_l2, divergence_max,
                                  source_add_ms, diffuse_ms, project_ms,
                                  advect_ms, boundary_ms, obstacle_ms,
                                  fade_ms, other_ms);
    }

    free_data();
    return 0;
}

int main(int argc, char **argv) {
    BenchmarkConfig bench_cfg;
    parse_benchmark_args(argc, argv, &bench_cfg);
    if (bench_cfg.enabled) {
        return run_benchmark_gpu3d(&bench_cfg);
    }

    glutInit(&argc, argv);
    printf("=== Stable Fluids 3D (GPU/CUDA wind tunnel viewer, M/Tab controls) ===\n");
    prompt_grid_size();
    printf("Grid: %d^3 interior cells, %.1f MB per scalar field\n",
           N, (double)(volume_count() * sizeof(float)) / (1024.0 * 1024.0));
    printf("Controls: m/tab camera-object mode, WASD/QE move selected, wheel speed, RMB look, b smoke, o flow, p pause, c clear, ESC quit\n\n");

    if (!allocate_data()) return 1;
    clear_data();

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(win_x, win_y);
    glutCreateWindow("Stable Fluids 3D - GPU/CUDA Volume Viewer [M/Tab Controls]");
    glClearColor(0, 0, 0, 1);

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);
    glutKeyboardUpFunc(keyboard_up);
    glutIgnoreKeyRepeat(1);
    last_fps_time = now_seconds();

    glutMainLoop();
    return 0;
}
