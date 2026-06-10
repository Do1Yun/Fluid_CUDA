// 2D Stable Fluids - GPU-only GLUT viewer
// Based on Jos Stam, "Real-Time Fluid Dynamics for Games" (GDC 2003)

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L  // for clock_gettime, CLOCK_MONOTONIC
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
#include <cuda_runtime.h>

#ifdef __APPLE__
  #include <GLUT/glut.h>
#else
  #include <GL/glut.h>
#endif

#include "solver.h"

#define SIZE 1024

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA ERROR %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err__));          \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

static int   N    = SIZE;
static float dt   = 0.1f;
static float diff = 0.0f;
static float visc = 0.0f;

static float dissipation = 0.995f;
static float force  = 5.0f;
static float source = 100.0f;
static float auto_smoke_velocity = -0.75f;
static int brush_cells_divisor = 64;
static float smoke_color_speed_scale = 0.55f;
static float velocity_vis_scale = 0.70f;
static int velocity_arrow_divisor = 24;

static float *h_u, *h_v, *h_u_prev, *h_v_prev;
static float *h_dens, *h_dens_prev;
static unsigned char *h_density_pixels;
static unsigned char *h_solid;

static float *d_u, *d_v, *d_u_prev, *d_v_prev;
static float *d_dens, *d_dens_prev;
static unsigned char *d_solid;
static unsigned char *d_density_pixels;
static GLuint density_tex = 0;

static int win_x = 512, win_y = 512;
static int mouse_down[3];
static int omx, omy, mx, my;
static int paused = 0;
static int draw_velocity = 1;
static int auto_smoke = 1;
static int obstacle_mode = 0;
static int obstacle_dirty = 0;
static int solid_active = 0;
static int solid_cell_count = 0;
static int velocity_host_valid = 0;

static int   frame_count   = 0;
static double fps_accum    = 0.0;
static double step_accum   = 0.0;
static double last_fps_time = 0.0;
static double current_fps  = 0.0;
static double current_step_ms = 0.0;

typedef struct BenchmarkConfig {
    int enabled;
    int frames;
    int warmup;
    int grid_size;
    char tag[32];
    int save_csv;
} BenchmarkConfig;

#define IX(i, j) ((i) + (N + 2) * (j))
#define SWAP(x0, x) { float *tmp = x0; x0 = x; x = tmp; }

static double now_seconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}

__device__ static unsigned char float_to_byte_device(float x) {
    x = fminf(fmaxf(x, 0.0f), 1.0f);
    return (unsigned char)(x * 255.0f + 0.5f);
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

__global__ void density_to_rgba_kernel(int N,
                                       const float *__restrict__ dens,
                                       const float *__restrict__ u,
                                       const float *__restrict__ v,
                                       const unsigned char *__restrict__ solid,
                                       unsigned char *__restrict__ pixels,
                                       float color_speed_scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= N && j <= N) {
        int idx = IX(i, j);
        int out = ((j - 1) * N + (i - 1)) * 4;
        if (solid && solid[idx]) {
            pixels[out + 0] = 26;
            pixels[out + 1] = 30;
            pixels[out + 2] = 34;
            pixels[out + 3] = 255;
            return;
        }

        float visual_dens = 1.0f - expf(-dens[idx] * 0.18f);
        visual_dens = fminf(fmaxf(visual_dens, 0.0f), 1.0f);

        float speed = sqrtf(u[idx] * u[idx] + v[idx] * v[idx]);
        float t = 1.0f - expf(-speed * color_speed_scale);
        t = fminf(t, 1.0f);

        float shade = visual_dens * (0.72f + 0.28f * t);
        float tint_r, tint_g, tint_b;
        hsv_to_rgb_device(0.58f - 0.48f * t, 0.78f, 1.0f,
                          &tint_r, &tint_g, &tint_b);

        float tint_strength = 0.10f + 0.42f * t;
        float r = shade * ((1.0f - tint_strength) + tint_strength * tint_r);
        float g = shade * ((1.0f - tint_strength) + tint_strength * tint_g);
        float b = shade * ((1.0f - tint_strength) + tint_strength * tint_b);
        pixels[out + 0] = float_to_byte_device(r);
        pixels[out + 1] = float_to_byte_device(g);
        pixels[out + 2] = float_to_byte_device(b);
        pixels[out + 3] = 255;
    }
}

__global__ void add_brush_source_kernel(int N,
                                        float *__restrict__ dens,
                                        float *__restrict__ u,
                                        float *__restrict__ v,
                                        int center_i, int center_j,
                                        int radius,
                                        float inv_radius2,
                                        float du, float dv,
                                        float source,
                                        int add_velocity,
                                        int add_density) {
    int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = center_i - radius + local_i;
    int j = center_j - radius + local_j;
    if (local_i > radius * 2 || local_j > radius * 2 ||
        i < 1 || i > N || j < 1 || j > N) {
        return;
    }

    int dx = i - center_i;
    int dy = j - center_j;
    float dist2 = (float)(dx * dx + dy * dy);
    if (dist2 > (float)(radius * radius)) return;

    float falloff = 1.0f - dist2 * inv_radius2;
    int idx = IX(i, j);
    if (add_velocity) {
        u[idx] += du * falloff;
        v[idx] += dv * falloff;
    }
    if (add_density) {
        dens[idx] += source * falloff;
    }
}

__global__ void add_auto_smoke_kernel(int N,
                                      float *__restrict__ dens,
                                      float *__restrict__ v,
                                      int radius,
                                      float inv_radius2,
                                      float source,
                                      float vertical_velocity) {
    int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_j = blockIdx.y * blockDim.y + threadIdx.y;
    int center_i = (N + 2) / 2;
    int center_j = (N + 2) / 2;
    int i = center_i - radius + local_i;
    int j = center_j - radius + local_j;
    if (local_i > radius * 2 || local_j > radius * 2 ||
        i < 1 || i > N || j < 1 || j > N) {
        return;
    }

    int dx = i - center_i;
    int dy = j - center_j;
    float dist2 = (float)(dx * dx + dy * dy);
    if (dist2 > (float)(radius * radius)) return;

    float falloff = 1.0f - dist2 * inv_radius2;
    int idx = IX(i, j);
    dens[idx] += source * falloff;
    v[idx] += vertical_velocity * falloff;
}

__global__ void zero_solid_fields_kernel(int N,
                                         float *__restrict__ dens,
                                         float *__restrict__ u,
                                         float *__restrict__ v,
                                         const unsigned char *__restrict__ solid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N + 2 && j < N + 2) {
        int idx = IX(i, j);
        if (solid[idx]) {
            dens[idx] = 0.0f;
            u[idx] = 0.0f;
            v[idx] = 0.0f;
        }
    }
}

static unsigned char *active_solid_device(void) {
    return solid_active ? d_solid : NULL;
}

static void clear_data(void) {
    int size = (N + 2) * (N + 2);
    int bytes = size * sizeof(float);
    memset(h_u, 0, bytes);
    memset(h_v, 0, bytes);
    memset(h_u_prev, 0, bytes);
    memset(h_v_prev, 0, bytes);
    memset(h_dens, 0, bytes);
    memset(h_dens_prev, 0, bytes);
    memset(h_density_pixels, 0, (size_t)N * N * 4);

    CUDA_CHECK(cudaMemset(d_u, 0, bytes));
    CUDA_CHECK(cudaMemset(d_v, 0, bytes));
    CUDA_CHECK(cudaMemset(d_u_prev, 0, bytes));
    CUDA_CHECK(cudaMemset(d_v_prev, 0, bytes));
    CUDA_CHECK(cudaMemset(d_dens, 0, bytes));
    CUDA_CHECK(cudaMemset(d_dens_prev, 0, bytes));
    CUDA_CHECK(cudaMemset(d_density_pixels, 0, (size_t)N * N * 4));
    velocity_host_valid = 0;
}

static void clear_obstacles(void) {
    int size = (N + 2) * (N + 2);
    memset(h_solid, 0, (size_t)size);
    CUDA_CHECK(cudaMemset(d_solid, 0, (size_t)size));
    obstacle_dirty = 0;
    solid_active = 0;
    solid_cell_count = 0;
}

static int allocate_data(void) {
    int size = (N + 2) * (N + 2);
    int bytes = size * sizeof(float);
    int mask_bytes = size * (int)sizeof(unsigned char);
    
    h_u         = (float *)malloc(bytes);
    h_v         = (float *)malloc(bytes);
    h_u_prev    = (float *)malloc(bytes);
    h_v_prev    = (float *)malloc(bytes);
    h_dens      = (float *)malloc(bytes);
    h_dens_prev = (float *)malloc(bytes);
    h_density_pixels = (unsigned char *)malloc((size_t)N * N * 4);
    h_solid = (unsigned char *)malloc(mask_bytes);
    if (!h_u || !h_v || !h_u_prev || !h_v_prev || !h_dens || !h_dens_prev ||
        !h_density_pixels || !h_solid) {
        fprintf(stderr, "ERROR: out of memory\n");
        return 0;
    }

    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_u_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_v_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_dens, bytes));
    CUDA_CHECK(cudaMalloc(&d_dens_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_solid, mask_bytes));
    CUDA_CHECK(cudaMalloc(&d_density_pixels, (size_t)N * N * 4));

    init_solver(N);
    return 1;
}

static void free_data(void) {
    free(h_u); free(h_v); free(h_u_prev); free(h_v_prev);
    free(h_dens); free(h_dens_prev);
    free(h_density_pixels);
    free(h_solid);
    if (density_tex) {
        glDeleteTextures(1, &density_tex);
        density_tex = 0;
    }
    
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_u_prev));
    CUDA_CHECK(cudaFree(d_v_prev));
    CUDA_CHECK(cudaFree(d_dens));
    CUDA_CHECK(cudaFree(d_dens_prev));
    CUDA_CHECK(cudaFree(d_solid));
    CUDA_CHECK(cudaFree(d_density_pixels));
    
    free_solver();
}

static void paint_obstacle_at(int erase) {
    int view_w = win_x > 0 ? win_x : 1;
    int view_h = win_y > 0 ? win_y : 1;
    int i = (int)((mx / (float)view_w) * N + 1);
    int j = (int)(((view_h - my) / (float)view_h) * N + 1);
    if (i < 1 || i > N || j < 1 || j > N) return;

    int radius = N / brush_cells_divisor;
    if (radius < 2) radius = 2;
    float inv_radius2 = 1.0f / (float)(radius * radius);
    for (int jj = j - radius; jj <= j + radius; jj++) {
        if (jj < 1 || jj > N) continue;
        for (int ii = i - radius; ii <= i + radius; ii++) {
            if (ii < 1 || ii > N) continue;
            int dx = ii - i;
            int dy = jj - j;
            float dist2 = (float)(dx * dx + dy * dy);
            if (dist2 > (float)(radius * radius)) continue;
            if (1.0f - dist2 * inv_radius2 <= 0.0f) continue;
            int idx = IX(ii, jj);
            unsigned char next = erase ? 0 : 1;
            if (h_solid[idx] != next) {
                solid_cell_count += next ? 1 : -1;
                if (solid_cell_count < 0) solid_cell_count = 0;
                h_solid[idx] = next;
            }
            h_u[idx] = h_v[idx] = h_dens[idx] = 0.0f;
        }
    }
    solid_active = (solid_cell_count > 0);
    obstacle_dirty = 1;
    velocity_host_valid = 0;
}

static void launch_auto_smoke_source(void) {
    int radius = N / brush_cells_divisor;
    if (radius < 2) radius = 2;
    float inv_radius2 = 1.0f / (float)(radius * radius);
    int side = radius * 2 + 1;
    dim3 threads(16, 16);
    dim3 blocks((side + 15) / 16, (side + 15) / 16);
    add_auto_smoke_kernel<<<blocks, threads>>>(N, d_dens_prev, d_v_prev,
                                               radius, inv_radius2, source,
                                               auto_smoke_velocity);
    CUDA_CHECK(cudaPeekAtLastError());
}

static void launch_brush_source(int i, int j, int local_mx, int local_omx) {
    int radius = N / brush_cells_divisor;
    if (radius < 2) radius = 2;
    float inv_radius2 = 1.0f / (float)(radius * radius);
    float du = force * (local_mx - local_omx);
    float dv = force * (omy - my);
    int side = radius * 2 + 1;
    dim3 threads(16, 16);
    dim3 blocks((side + 15) / 16, (side + 15) / 16);
    add_brush_source_kernel<<<blocks, threads>>>(N, d_dens_prev, d_u_prev, d_v_prev,
                                                i, j, radius, inv_radius2,
                                                du, dv, source,
                                                mouse_down[0], mouse_down[2]);
    CUDA_CHECK(cudaPeekAtLastError());
}

static void prepare_device_sources_from_ui(void) {
    int bytes = (N + 2) * (N + 2) * sizeof(float);
    CUDA_CHECK(cudaMemset(d_dens_prev, 0, bytes));
    CUDA_CHECK(cudaMemset(d_u_prev, 0, bytes));
    CUDA_CHECK(cudaMemset(d_v_prev, 0, bytes));

    if (obstacle_mode) {
        if (mouse_down[0]) paint_obstacle_at(0);
        if (mouse_down[2]) paint_obstacle_at(1);
        return;
    }

    if (mouse_down[0] || mouse_down[2]) {
        int view_w = win_x > 0 ? win_x : 1;
        int view_h = win_y > 0 ? win_y : 1;
        int local_mx = mx;
        int local_omx = omx;
        if (local_mx < 0) local_mx = 0;
        if (local_mx >= view_w) local_mx = view_w - 1;
        if (local_omx < 0) local_omx = 0;
        if (local_omx >= view_w) local_omx = view_w - 1;

        int i = (int)((local_mx / (float)view_w) * N + 1);
        int j = (int)(((view_h - my) / (float)view_h) * N + 1);
        if (i >= 1 && i <= N && j >= 1 && j <= N) {
            launch_brush_source(i, j, local_mx, local_omx);
        }
        omx = mx;
        omy = my;
    }

    if (auto_smoke) {
        launch_auto_smoke_source();
    }
}

static void ensure_density_texture(void) {
    if (density_tex) return;

    glGenTextures(1, &density_tex);
    glBindTexture(GL_TEXTURE_2D, density_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, N, N, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

static void update_density_texture_from_device(void) {
    ensure_density_texture();

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    density_to_rgba_kernel<<<blocks, threads>>>(N, d_dens, d_u, d_v,
                                                active_solid_device(),
                                                d_density_pixels,
                                                smoke_color_speed_scale);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaMemcpy(h_density_pixels, d_density_pixels,
                          (size_t)N * N * 4, cudaMemcpyDeviceToHost));

    glBindTexture(GL_TEXTURE_2D, density_tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, N, N,
                    GL_RGBA, GL_UNSIGNED_BYTE, h_density_pixels);
}

static void draw_density_field(void) {
    update_density_texture_from_device();

    glColor3f(1.0f, 1.0f, 1.0f);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, density_tex);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 1.0f);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

static void draw_velocity_field(const float *u, const float *v) {
    float h = 1.0f / N;
    int stride = N / velocity_arrow_divisor;
    if (stride < 4) stride = 4;
    float spacing = stride * h;

    glLineWidth(2.0f);
    glBegin(GL_LINES);
    for (int i = stride / 2 + 1; i <= N; i += stride) {
        float x = (i - 0.5f) * h;
        for (int j = stride / 2 + 1; j <= N; j += stride) {
            float y = (j - 0.5f) * h;
            float vx = 0.0f;
            float vy = 0.0f;
            int count = 0;
            int half = stride / 2;
            for (int jj = j - half; jj <= j + half; jj++) {
                if (jj < 1 || jj > N) continue;
                for (int ii = i - half; ii <= i + half; ii++) {
                    if (ii < 1 || ii > N) continue;
                    vx += u[IX(ii, jj)];
                    vy += v[IX(ii, jj)];
                    count++;
                }
            }
            if (count > 0) {
                vx /= (float)count;
                vy /= (float)count;
            }

            float speed = sqrtf(vx * vx + vy * vy);
            if (speed < 0.00015f) continue;

            float magnitude = 1.0f - expf(-speed * velocity_vis_scale);
            float len = spacing * (0.22f + 1.05f * magnitude);
            float dir_x = vx / speed;
            float dir_y = vy / speed;
            float end_x = x + dir_x * len;
            float end_y = y + dir_y * len;
            float head_len = len * 0.30f;
            float head_w = len * 0.18f;
            float px = -dir_y;
            float py = dir_x;
            float intensity = 1.0f - expf(-speed * 0.70f);
            if (intensity > 1.0f) intensity = 1.0f;

            glColor3f(0.10f + 0.90f * intensity,
                      0.85f + 0.15f * intensity,
                      1.00f - 0.65f * intensity);
            glVertex2f(x, y);
            glVertex2f(end_x, end_y);

            glVertex2f(end_x, end_y);
            glVertex2f(end_x - dir_x * head_len + px * head_w,
                       end_y - dir_y * head_len + py * head_w);

            glVertex2f(end_x, end_y);
            glVertex2f(end_x - dir_x * head_len - px * head_w,
                       end_y - dir_y * head_len - py * head_w);
        }
    }
    glEnd();
    glLineWidth(1.0f);
}

static void draw_text(float x, float y, const char *str) {
    glColor3f(1.0f, 1.0f, 0.2f);
    glRasterPos2f(x, y);
    for (const char *c = str; *c; c++)
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *c);
}

static void copy_velocity_to_host_if_needed(void) {
    if (velocity_host_valid) return;
    int bytes = (N + 2) * (N + 2) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(h_u, d_u, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost));
    velocity_host_valid = 1;
}

static void draw_panel(int x, int w, const char *name, double step_ms) {
    glViewport(x, 0, w, win_y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    draw_density_field();
    if (draw_velocity) {
        copy_velocity_to_host_if_needed();
        draw_velocity_field(h_u, h_v);
    }

    char buf[160];
    snprintf(buf, sizeof(buf), "%s  %.2f ms%s", name, step_ms,
             paused ? "  [PAUSED]" : "");
    draw_text(0.01f, 0.93f, buf);
}

static void display(void) {
    glClear(GL_COLOR_BUFFER_BIT);

    draw_panel(0, win_x, "GPU", current_step_ms);

    char buf[128];
    snprintf(buf, sizeof(buf), "N=%d  FPS=%.1f  v: vectors  a: auto %s  o: obstacle %s",
             N, current_fps, auto_smoke ? "on" : "off",
             obstacle_mode ? "paint" : "off");
    glViewport(0, 0, win_x, win_y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    draw_text(0.01f, 0.97f, buf);
    draw_text(0.01f, 0.02f,
              obstacle_mode
              ? "Obstacle: LMB draw | RMB erase | o: fluid mode | x: clear walls | ESC: quit"
              : "LMB: velocity | RMB: smoke | o: walls | a: auto | c: clear | p: pause | v: vectors | ESC: quit");

    glutSwapBuffers();
}

static void idle(void) {
    if (!paused) {
        prepare_device_sources_from_ui();

        int mask_bytes = (N + 2) * (N + 2) * (int)sizeof(unsigned char);
        if (obstacle_dirty) {
            CUDA_CHECK(cudaMemcpy(d_solid, h_solid, mask_bytes, cudaMemcpyHostToDevice));
            if (solid_active) {
                dim3 threads(16, 16);
                dim3 blocks((N + 2 + 15) / 16, (N + 2 + 15) / 16);
                zero_solid_fields_kernel<<<blocks, threads>>>(N, d_dens, d_u, d_v, d_solid);
                CUDA_CHECK(cudaPeekAtLastError());
            }
            obstacle_dirty = 0;
        }
        
        double step_t0 = now_seconds();
        vel_step(N, d_u, d_v, d_u_prev, d_v_prev, visc, dt, active_solid_device());
        dens_step(N, d_dens, d_dens_prev, d_u, d_v, diff, dt, active_solid_device());
        
        fade_fields(N, d_dens, d_u, d_v, dissipation, 0.99f, active_solid_device());
        
        CUDA_CHECK(cudaDeviceSynchronize());
        double step_t1 = now_seconds();
        velocity_host_valid = 0;

        step_accum += (step_t1 - step_t0);
        fps_accum += (step_t1 - step_t0);
        frame_count++;

        double now = now_seconds();
        if (now - last_fps_time >= 0.5) {
            current_fps = frame_count / (fps_accum > 0 ? fps_accum : 1.0);
            current_step_ms = (step_accum / frame_count) * 1000.0;
            printf("N=%d  GPU=%.3f ms  fps=%.1f\n",
                   N, current_step_ms, current_fps);
            frame_count = 0;
            fps_accum = 0.0;
            step_accum = 0.0;
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
    omx = mx = x;
    omy = my = y;
    if (button == GLUT_LEFT_BUTTON)   mouse_down[0] = (state == GLUT_DOWN);
    if (button == GLUT_MIDDLE_BUTTON) mouse_down[1] = (state == GLUT_DOWN);
    if (button == GLUT_RIGHT_BUTTON)  mouse_down[2] = (state == GLUT_DOWN);
}

static void motion(int x, int y) {
    mx = x;
    my = y;
}

static void keyboard(unsigned char key, int x, int y) {
    (void)x; (void)y;
    switch (key) {
        case 'c': case 'C': clear_data(); break;
        case 'q': case 'Q': case 27: free_data(); exit(0);
        case 'p': case 'P': paused = !paused; break;
        case 'v': case 'V': draw_velocity = !draw_velocity; break;
        case 'a': case 'A': auto_smoke = !auto_smoke; break;
        case 'o': case 'O': obstacle_mode = !obstacle_mode; break;
        case 'x': case 'X': clear_obstacles(); break;
    }
}

static void prompt_grid_size(void) {
    char line[64];
    printf("Grid size N [default %d, suggested 256/512/1024]: ", N);
    fflush(stdout);

    if (!fgets(line, sizeof(line), stdin)) {
        return;
    }

    if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') {
        return;
    }

    int value = atoi(line);
    if (value < 16) {
        printf("N is too small; using 16.\n");
        value = 16;
    } else if (value > 2048) {
        printf("N is too large for this viewer; using 2048.\n");
        value = 2048;
    }
    N = value;
}

static void init_benchmark_config(BenchmarkConfig *cfg) {
    cfg->enabled = 0;
    cfg->frames = 240;
    cfg->warmup = 30;
    cfg->grid_size = 256;
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
    if (cfg->grid_size > 2048) cfg->grid_size = 2048;
}

static int clamp_cell_gpu2d(int value) {
    if (value < 1) return 1;
    if (value > N) return N;
    return value;
}

static void benchmark_source_params_2d(int frame, int pass,
                                       int *center_i, int *center_j,
                                       float *density_value,
                                       float *du, float *dv) {
    float phase = ((float)frame + 1.0f) * 0.071f + (float)pass * 2.173f;
    float sx = (pass == 0) ? sinf(phase) : cosf(phase * 0.83f);
    float sy = (pass == 0) ? cosf(phase * 0.91f) : sinf(phase * 1.17f);
    float x = 0.50f + ((pass == 0) ? 0.23f : -0.19f) * sx;
    float y = 0.50f + ((pass == 0) ? 0.18f : 0.22f) * sy;

    *center_i = clamp_cell_gpu2d((int)(x * (float)N) + 1);
    *center_j = clamp_cell_gpu2d((int)(y * (float)N) + 1);
    *density_value = source * ((pass == 0) ? 1.00f : 0.72f);
    *du = force * (0.90f * cosf(phase * 1.31f) + 0.25f * sinf(phase * 0.47f));
    *dv = force * (0.90f * sinf(phase * 1.07f) - 0.20f * cosf(phase * 0.73f));
}

static void launch_benchmark_blob_2d(int center_i, int center_j, int radius,
                                     float density_value, float du, float dv) {
    float inv_radius2 = 1.0f / (float)(radius * radius);
    int side = radius * 2 + 1;
    dim3 threads(16, 16);
    dim3 blocks((side + 15) / 16, (side + 15) / 16);
    add_brush_source_kernel<<<blocks, threads>>>(N, d_dens_prev, d_u_prev, d_v_prev,
                                                center_i, center_j, radius,
                                                inv_radius2, du, dv,
                                                density_value, 1, 1);
    CUDA_CHECK(cudaPeekAtLastError());
}

static void prepare_benchmark_sources_gpu2d(int frame) {
    int bytes = (N + 2) * (N + 2) * (int)sizeof(float);
    CUDA_CHECK(cudaMemset(d_dens_prev, 0, bytes));
    CUDA_CHECK(cudaMemset(d_u_prev, 0, bytes));
    CUDA_CHECK(cudaMemset(d_v_prev, 0, bytes));

    int radius = N / 32;
    if (radius < 3) radius = 3;
    for (int pass = 0; pass < 2; pass++) {
        int center_i, center_j;
        float density_value, du, dv;
        benchmark_source_params_2d(frame, pass, &center_i, &center_j,
                                   &density_value, &du, &dv);
        launch_benchmark_blob_2d(center_i, center_j, radius,
                                 density_value, du, dv);
    }
}

static void compute_benchmark_metrics_gpu2d(double *density_sum,
                                            float *density_max,
                                            double *velocity_l2,
                                            double *divergence_l2,
                                            float *divergence_max) {
    int bytes = (N + 2) * (N + 2) * (int)sizeof(float);
    CUDA_CHECK(cudaMemcpy(h_dens, d_dens, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_u, d_u, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost));

    double density_accum = 0.0;
    double velocity_accum = 0.0;
    double divergence_accum = 0.0;
    float max_density = 0.0f;
    float max_divergence = 0.0f;
    int count = 0;

    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            int idx = IX(i, j);
            float dens_value = h_dens[idx];
            float speed2 = h_u[idx] * h_u[idx] + h_v[idx] * h_v[idx];
            float div = 0.5f * (float)N *
                        (h_u[IX(i + 1, j)] - h_u[IX(i - 1, j)] +
                         h_v[IX(i, j + 1)] - h_v[IX(i, j - 1)]);
            float abs_div = fabsf(div);

            density_accum += (double)dens_value;
            velocity_accum += (double)speed2;
            divergence_accum += (double)div * (double)div;
            if (dens_value > max_density) max_density = dens_value;
            if (abs_div > max_divergence) max_divergence = abs_div;
            count++;
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

static void write_benchmark_csv_gpu2d(const char *mode,
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
                                      float divergence_max) {
    char timestamp[32];
    const char *filename = (strcmp(tag, "scaling") == 0) ?
        "benchmark_scaling_2d.csv" : "benchmark_2d.csv";
    make_benchmark_timestamp(timestamp, sizeof(timestamp));
    long long cells = (long long)grid_size * (long long)grid_size;
    double scalar_field_mb = ((double)(grid_size + 2) * (double)(grid_size + 2) *
                              (double)sizeof(float)) / (1024.0 * 1024.0);
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
        fseek(probe, 0, SEEK_END);
        need_header = (ftell(probe) == 0);
        fclose(probe);
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
        fseek(probe, 0, SEEK_END);
        need_header = (ftell(probe) == 0);
        fclose(probe);
    }
    FILE *fp = fopen(csv_path, "ab");
#endif
    if (!fp) {
        fprintf(stderr, "WARNING: could not write benchmark CSV: %s\n", filename);
        return;
    }

    if (need_header) {
        fprintf(fp, "timestamp,task,mode,input_id,N,dimension,cells,scalar_field_mb,warmup,frames,source_ms,step_ms,total_ms,fps,mcells_per_sec,ns_per_cell,density_sum,density_max,velocity_l2,divergence_l2,divergence_max\n");
    }
    fprintf(fp, "%s,2d,%s,%s,%d,2,%lld,%.6f,%d,%d,%.6f,%.6f,%.6f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6e,%.6e\n",
            timestamp, mode, input_id, grid_size, cells, scalar_field_mb,
            warmup_frames, measured_frames,
            source_ms, step_ms, total_ms, fps,
            mcells_per_sec, ns_per_cell,
            density_sum, density_max, velocity_l2,
            divergence_l2, divergence_max);
    fclose(fp);

    printf("benchmark_csv,benchmark_results/%s\n", filename);
}

static int run_benchmark_gpu2d(const BenchmarkConfig *cfg) {
    N = cfg->grid_size;
    printf("=== Stable Fluids 2D (GPU/CUDA benchmark) ===\n");
    printf("Benchmark input: deterministic_2d_v1, no GLUT, no user input\n");
    printf("Grid: N=%d, warmup=%d, frames=%d\n", N, cfg->warmup, cfg->frames);

    if (!allocate_data()) return 1;
    clear_obstacles();
    clear_data();

    int total_frames = cfg->warmup + cfg->frames;
    double source_seconds = 0.0;
    double step_seconds = 0.0;

    for (int frame = 0; frame < total_frames; frame++) {
        double source_t0 = now_seconds();
        prepare_benchmark_sources_gpu2d(frame);
        CUDA_CHECK(cudaDeviceSynchronize());
        double source_t1 = now_seconds();

        double step_t0 = now_seconds();
        vel_step(N, d_u, d_v, d_u_prev, d_v_prev, visc, dt, NULL);
        dens_step(N, d_dens, d_dens_prev, d_u, d_v, diff, dt, NULL);
        fade_fields(N, d_dens, d_u, d_v, dissipation, 0.99f, NULL);
        CUDA_CHECK(cudaDeviceSynchronize());
        double step_t1 = now_seconds();

        if (frame >= cfg->warmup) {
            source_seconds += source_t1 - source_t0;
            step_seconds += step_t1 - step_t0;
        }
    }

    double density_sum;
    double velocity_l2;
    double divergence_l2;
    float density_max;
    float divergence_max;
    compute_benchmark_metrics_gpu2d(&density_sum, &density_max,
                                    &velocity_l2, &divergence_l2,
                                    &divergence_max);

    double source_ms = source_seconds * 1000.0 / (double)cfg->frames;
    double step_ms = step_seconds * 1000.0 / (double)cfg->frames;
    double total_ms = source_ms + step_ms;
    double fps = (total_ms > 0.0) ? 1000.0 / total_ms : 0.0;

    printf("benchmark_header,mode,N,warmup,frames,source_ms,step_ms,total_ms,fps,density_sum,density_max,velocity_l2,divergence_l2,divergence_max\n");
    printf("benchmark_result,gpu2d,%d,%d,%d,%.6f,%.6f,%.6f,%.3f,%.6f,%.6f,%.6f,%.6e,%.6e\n",
           N, cfg->warmup, cfg->frames,
           source_ms, step_ms, total_ms, fps,
           density_sum, density_max, velocity_l2,
           divergence_l2, divergence_max);
    if (cfg->save_csv) {
        write_benchmark_csv_gpu2d("gpu2d", "deterministic_2d_v1",
                                  cfg->tag, N, cfg->warmup, cfg->frames,
                                  source_ms, step_ms, total_ms, fps,
                                  density_sum, density_max, velocity_l2,
                                  divergence_l2, divergence_max);
    }

    free_data();
    return 0;
}

int main(int argc, char **argv) {
    BenchmarkConfig bench_cfg;
    parse_benchmark_args(argc, argv, &bench_cfg);
    if (bench_cfg.enabled) {
        return run_benchmark_gpu2d(&bench_cfg);
    }

    glutInit(&argc, argv);

    printf("=== Stable Fluids 2D (GPU/CUDA) ===\n");
    prompt_grid_size();
    printf("Grid: N=%d (interior), total %dx%d cells\n",
           N, N + 2, N + 2);
    printf("Controls:\n");
    printf("  Left mouse drag  : add velocity\n");
    printf("  Right mouse drag : add density (paint)\n");
    printf("  c                : clear\n");
    printf("  p                : pause\n");
    printf("  v                : toggle velocity field\n");
    printf("  a                : toggle automatic smoke source\n");
    printf("  o                : obstacle paint mode (LMB draw, RMB erase)\n");
    printf("  x                : clear obstacles\n");
    printf("  q / ESC          : quit\n\n");

    if (!allocate_data()) return 1;
    clear_obstacles();
    clear_data();

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(win_x, win_y);
    glutCreateWindow("Stable Fluids 2D - GPU/CUDA");

    glClearColor(0, 0, 0, 1);

    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
    glutDisplayFunc(display);

    last_fps_time = now_seconds();

    glutMainLoop();
    return 0;
}
