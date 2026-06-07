// 3D Stable Fluids - GPU slice viewer

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
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
static float dissipation = 0.992f;
static float source = 180.0f;
static float force = 3.5f;
static float color_speed_scale = 0.55f;

static float *h_dens, *h_dens_prev;
static float *h_u, *h_v, *h_w, *h_u_prev, *h_v_prev, *h_w_prev;
static float *d_dens, *d_dens_prev;
static float *d_u, *d_v, *d_w, *d_u_prev, *d_v_prev, *d_w_prev;

static int win_x = 900, win_y = 760;
static int mx, my, omx, omy;
static int mouse_down[3];
static int paused = 0;
static float camera_slice_offset = 0.0f;
static int show_vectors = 1;
static int show_slice = 1;
static int show_volume = 1;
static int auto_source = 0;
static float last_inject_pos[3] = {0.0f, 0.0f, 0.0f};
static int has_inject_pos = 0;

static float cam_x = 0.0f, cam_y = 0.0f, cam_z = 2.2f;
static float cam_yaw = 0.0f, cam_pitch = 0.0f;
static float cam_speed = 0.08f;
static int right_look = 0;
static int keys[256];

static int frame_count = 0;
static double accum_time = 0.0;
static double last_fps_time = 0.0;
static double current_ms = 0.0;
static double current_fps = 0.0;

#define IX3(i, j, k) ((i) + (N + 2) * ((j) + (N + 2) * (k)))

static float cell_coord(int index);

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

static void camera_plane_center(float *center) {
    float forward[3], right[3], up[3];
    camera_basis(forward, right, up);

    center[0] = forward[0] * camera_slice_offset;
    center[1] = forward[1] * camera_slice_offset;
    center[2] = forward[2] * camera_slice_offset;
}

static int screen_to_camera_plane(int x, int y, float *out) {
    float forward[3], right[3], up[3], center[3];
    camera_basis(forward, right, up);
    camera_plane_center(center);

    float aspect = (win_y > 0) ? (float)win_x / (float)win_y : 1.0f;
    float fov = 58.0f * 3.14159265f / 180.0f;
    float tan_half = tanf(fov * 0.5f);
    float sx = ((float)x / (float)(win_x > 0 ? win_x : 1)) * 2.0f - 1.0f;
    float sy = 1.0f - ((float)y / (float)(win_y > 0 ? win_y : 1)) * 2.0f;

    float ray[3] = {
        forward[0] + right[0] * sx * tan_half * aspect + up[0] * sy * tan_half,
        forward[1] + right[1] * sx * tan_half * aspect + up[1] * sy * tan_half,
        forward[2] + right[2] * sx * tan_half * aspect + up[2] * sy * tan_half
    };
    float inv_len = 1.0f / sqrtf(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]);
    ray[0] *= inv_len;
    ray[1] *= inv_len;
    ray[2] *= inv_len;

    float denom = ray[0] * forward[0] + ray[1] * forward[1] + ray[2] * forward[2];
    if (fabsf(denom) <= 0.0001f) return 0;
    float numer = ((center[0] - cam_x) * forward[0] +
                   (center[1] - cam_y) * forward[1] +
                   (center[2] - cam_z) * forward[2]);
    float t = numer / denom;
    if (t <= 0.0f) return 0;

    out[0] = cam_x + ray[0] * t;
    out[1] = cam_y + ray[1] * t;
    out[2] = cam_z + ray[2] * t;
    return 1;
}

static int world_to_cell(float p) {
    int cell = (int)((p + 0.5f) * (float)N) + 1;
    if (cell < 1) cell = 1;
    if (cell > N) cell = N;
    return cell;
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

static void clear_host_sources(void) {
    size_t bytes = volume_count() * sizeof(float);
    memset(h_dens_prev, 0, bytes);
    memset(h_u_prev, 0, bytes);
    memset(h_v_prev, 0, bytes);
    memset(h_w_prev, 0, bytes);
}

static void clear_data(void) {
    size_t bytes = volume_count() * sizeof(float);
    memset(h_dens, 0, bytes);
    memset(h_u, 0, bytes);
    memset(h_v, 0, bytes);
    memset(h_w, 0, bytes);
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
    h_dens = (float *)malloc(bytes);
    h_dens_prev = (float *)malloc(bytes);
    h_u = (float *)malloc(bytes);
    h_v = (float *)malloc(bytes);
    h_w = (float *)malloc(bytes);
    h_u_prev = (float *)malloc(bytes);
    h_v_prev = (float *)malloc(bytes);
    h_w_prev = (float *)malloc(bytes);
    if (!h_dens || !h_dens_prev || !h_u || !h_v || !h_w ||
        !h_u_prev || !h_v_prev || !h_w_prev) {
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

static void free_data(void) {
    free(h_dens); free(h_dens_prev);
    free(h_u); free(h_v); free(h_w);
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

static void hsv_to_rgb(float h, float s, float v, float *r, float *g, float *b) {
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

static void smoke_rgba(float dens, float u, float v, float w,
                       float alpha_scale,
                       float *r, float *g, float *b, float *a) {
    float visual_dens = 1.0f - expf(-dens * 0.18f);
    if (visual_dens < 0.0f) visual_dens = 0.0f;
    if (visual_dens > 1.0f) visual_dens = 1.0f;
    float speed = sqrtf(u * u + v * v + w * w);
    float t = 1.0f - expf(-speed * color_speed_scale);
    if (t > 1.0f) t = 1.0f;

    float tint_r, tint_g, tint_b;
    hsv_to_rgb(0.58f - 0.48f * t, 0.78f, 1.0f, &tint_r, &tint_g, &tint_b);
    float shade = visual_dens * (0.72f + 0.28f * t);
    float tint = 0.10f + 0.42f * t;
    *r = shade * ((1.0f - tint) + tint * tint_r);
    *g = shade * ((1.0f - tint) + tint * tint_g);
    *b = shade * ((1.0f - tint) + tint * tint_b);
    *a = visual_dens * alpha_scale;
    if (*a > 0.70f) *a = 0.70f;
}

static void set_smoke_color(float dens, float u, float v, float w, float alpha_scale) {
    float r, g, b, a;
    smoke_rgba(dens, u, v, w, alpha_scale, &r, &g, &b, &a);
    glColor4f(r, g, b, a);
}

static void add_spherical_source(float *dens, float *u, float *v, float *w) {
    int cx = N / 2 + 1;
    int cy = N / 2 + 1;
    int cz = N / 2 + 1;
    int radius = N / 10;
    if (radius < 3) radius = 3;
    float inv_r2 = 1.0f / (float)(radius * radius);

    for (int k = cz - radius; k <= cz + radius; k++) {
        if (k < 1 || k > N) continue;
        for (int j = cy - radius; j <= cy + radius; j++) {
            if (j < 1 || j > N) continue;
            for (int i = cx - radius; i <= cx + radius; i++) {
                if (i < 1 || i > N) continue;
                int dx = i - cx, dy = j - cy, dz = k - cz;
                float dist2 = (float)(dx * dx + dy * dy + dz * dz);
                if (dist2 > (float)(radius * radius)) continue;
                float falloff = 1.0f - dist2 * inv_r2;
                int idx = IX3(i, j, k);
                dens[idx] += source * falloff;
                v[idx] += 1.7f * falloff;
                w[idx] += 0.6f * sinf((float)frame_count * 0.08f) * falloff;
            }
        }
    }
}

static void get_from_ui(void) {
    clear_host_sources();
    if (auto_source) {
        add_spherical_source(h_dens_prev, h_u_prev, h_v_prev, h_w_prev);
    }

    if (!mouse_down[0]) return;
    float p[3];
    float forward[3], right[3], up[3];
    if (!screen_to_camera_plane(mx, my, p)) return;
    if (p[0] < -0.5f || p[0] > 0.5f ||
        p[1] < -0.5f || p[1] > 0.5f ||
        p[2] < -0.5f || p[2] > 0.5f) {
        return;
    }
    camera_basis(forward, right, up);
    last_inject_pos[0] = p[0];
    last_inject_pos[1] = p[1];
    last_inject_pos[2] = p[2];
    has_inject_pos = 1;

    int i = world_to_cell(p[0]);
    int j = world_to_cell(p[1]);
    int k = world_to_cell(p[2]);

    int radius = N / 7;
    if (radius < 5) radius = 5;
    float drag_x = (float)(mx - omx);
    float drag_y = (float)(omy - my);
    float impulse = force * 0.08f;
    float du = impulse * (right[0] * drag_x + up[0] * drag_y);
    float dv = impulse * (right[1] * drag_x + up[1] * drag_y);
    float dw = impulse * (right[2] * drag_x + up[2] * drag_y);

    float brush_radius = (float)radius / (float)N;
    float brush_depth = brush_radius * 0.95f;
    if (brush_depth < 5.0f / (float)N) brush_depth = 5.0f / (float)N;
    float inv_r2 = 1.0f / (brush_radius * brush_radius);
    float inv_depth2 = 1.0f / (brush_depth * brush_depth);
    for (int kk = k - radius; kk <= k + radius; kk++) {
        if (kk < 1 || kk > N) continue;
        for (int jj = j - radius; jj <= j + radius; jj++) {
            if (jj < 1 || jj > N) continue;
            for (int ii = i - radius; ii <= i + radius; ii++) {
                if (ii < 1 || ii > N) continue;
                float dx = cell_coord(ii) - p[0];
                float dy = cell_coord(jj) - p[1];
                float dz = cell_coord(kk) - p[2];
                float plane_x = dx * right[0] + dy * right[1] + dz * right[2];
                float plane_y = dx * up[0] + dy * up[1] + dz * up[2];
                float plane_z = dx * forward[0] + dy * forward[1] + dz * forward[2];
                float volume_r2 = plane_x * plane_x * inv_r2 +
                                  plane_y * plane_y * inv_r2 +
                                  plane_z * plane_z * inv_depth2;
                if (volume_r2 > 1.0f) continue;
                float falloff = 1.0f - volume_r2;
                falloff *= falloff;
                float depth_push = force * 0.12f * falloff;
                float swirl = force * 0.05f * falloff / brush_radius;
                int idx = IX3(ii, jj, kk);
                h_u_prev[idx] += du * falloff + forward[0] * depth_push + (right[0] * plane_y - up[0] * plane_x) * swirl;
                h_v_prev[idx] += dv * falloff + forward[1] * depth_push + (right[1] * plane_y - up[1] * plane_x) * swirl;
                h_w_prev[idx] += dw * falloff + forward[2] * depth_push + (right[2] * plane_y - up[2] * plane_x) * swirl;
                h_dens_prev[idx] += source * falloff;
            }
        }
    }
    omx = mx;
    omy = my;
}

static void update_camera(void) {
    float cy = cosf(cam_yaw), sy = sinf(cam_yaw);
    float fx = sy, fz = -cy;
    float rx = cy, rz = sy;
    if (keys['w'] || keys['W']) { cam_x += fx * cam_speed; cam_z += fz * cam_speed; }
    if (keys['s'] || keys['S']) { cam_x -= fx * cam_speed; cam_z -= fz * cam_speed; }
    if (keys['d'] || keys['D']) { cam_x += rx * cam_speed; cam_z += rz * cam_speed; }
    if (keys['a'] || keys['A']) { cam_x -= rx * cam_speed; cam_z -= rz * cam_speed; }
    if (keys['e'] || keys['E']) cam_y += cam_speed;
    if (keys['q'] || keys['Q']) cam_y -= cam_speed;
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
    float cp = cosf(cam_pitch);
    float fx = sinf(cam_yaw) * cp;
    float fy = -sinf(cam_pitch);
    float fz = -cosf(cam_yaw) * cp;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(58.0, aspect, 0.03, 30.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(cam_x, cam_y, cam_z,
              cam_x + fx, cam_y + fy, cam_z + fz,
              0.0, 1.0, 0.0);
}

static void draw_cube_bounds(void) {
    glDisable(GL_BLEND);
    glLineWidth(1.5f);
    glColor3f(0.28f, 0.34f, 0.40f);

    glBegin(GL_LINE_LOOP);
    glVertex3f(-0.5f, -0.5f, -0.5f); glVertex3f(0.5f, -0.5f, -0.5f);
    glVertex3f(0.5f, 0.5f, -0.5f); glVertex3f(-0.5f, 0.5f, -0.5f);
    glEnd();
    glBegin(GL_LINE_LOOP);
    glVertex3f(-0.5f, -0.5f, 0.5f); glVertex3f(0.5f, -0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, 0.5f); glVertex3f(-0.5f, 0.5f, 0.5f);
    glEnd();
    glBegin(GL_LINES);
    glVertex3f(-0.5f, -0.5f, -0.5f); glVertex3f(-0.5f, -0.5f, 0.5f);
    glVertex3f(0.5f, -0.5f, -0.5f); glVertex3f(0.5f, -0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, -0.5f); glVertex3f(0.5f, 0.5f, 0.5f);
    glVertex3f(-0.5f, 0.5f, -0.5f); glVertex3f(-0.5f, 0.5f, 0.5f);
    glEnd();

    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glColor3f(1.0f, 0.25f, 0.20f);
    glVertex3f(-0.58f, -0.58f, -0.58f); glVertex3f(-0.38f, -0.58f, -0.58f);
    glColor3f(0.35f, 1.0f, 0.35f);
    glVertex3f(-0.58f, -0.58f, -0.58f); glVertex3f(-0.58f, -0.38f, -0.58f);
    glColor3f(0.35f, 0.55f, 1.0f);
    glVertex3f(-0.58f, -0.58f, -0.58f); glVertex3f(-0.58f, -0.58f, -0.38f);
    glEnd();
    glLineWidth(1.0f);
}

static void draw_input_plane_guide(void) {
    float forward[3], right[3], up[3];
    camera_basis(forward, right, up);
    float center[3];
    camera_plane_center(center);
    float s = 0.72f;

    glDisable(GL_BLEND);
    glLineWidth(1.0f);
    glColor3f(0.42f, 0.48f, 0.56f);
    glBegin(GL_LINE_LOOP);
    glVertex3f(center[0] + (-right[0] - up[0]) * s,
               center[1] + (-right[1] - up[1]) * s,
               center[2] + (-right[2] - up[2]) * s);
    glVertex3f(center[0] + ( right[0] - up[0]) * s,
               center[1] + ( right[1] - up[1]) * s,
               center[2] + ( right[2] - up[2]) * s);
    glVertex3f(center[0] + ( right[0] + up[0]) * s,
               center[1] + ( right[1] + up[1]) * s,
               center[2] + ( right[2] + up[2]) * s);
    glVertex3f(center[0] + (-right[0] + up[0]) * s,
               center[1] + (-right[1] + up[1]) * s,
               center[2] + (-right[2] + up[2]) * s);
    glEnd();

    if (has_inject_pos) {
        float m = 0.035f;
        glLineWidth(2.0f);
        glColor3f(1.0f, 0.95f, 0.25f);
        glBegin(GL_LINES);
        glVertex3f(last_inject_pos[0] - right[0] * m, last_inject_pos[1] - right[1] * m, last_inject_pos[2] - right[2] * m);
        glVertex3f(last_inject_pos[0] + right[0] * m, last_inject_pos[1] + right[1] * m, last_inject_pos[2] + right[2] * m);
        glVertex3f(last_inject_pos[0] - up[0] * m, last_inject_pos[1] - up[1] * m, last_inject_pos[2] - up[2] * m);
        glVertex3f(last_inject_pos[0] + up[0] * m, last_inject_pos[1] + up[1] * m, last_inject_pos[2] + up[2] * m);
        glEnd();
    }
}

static float cell_coord(int index) {
    return ((float)index - 0.5f) / (float)N - 0.5f;
}

static void draw_oriented_density_plane(float offset,
                                        float alpha_scale,
                                        int draw_samples,
                                        int draw_border) {
    float forward[3], right[3], up[3], center[3];
    camera_basis(forward, right, up);
    camera_plane_center(center);
    center[0] += forward[0] * offset;
    center[1] += forward[1] * offset;
    center[2] += forward[2] * offset;

    if (draw_samples) {
        int samples = (N > 96) ? 64 : N;
        float h = 1.0f / (float)samples;

        glBegin(GL_QUADS);
        for (int j = 0; j < samples; j++) {
            float y0 = (float)j * h - 0.5f;
            float y1 = y0 + h;
            for (int i = 0; i < samples; i++) {
                float x0 = (float)i * h - 0.5f;
                float x1 = x0 + h;
                float px[4] = {x0, x1, x1, x0};
                float py[4] = {y0, y0, y1, y1};
                for (int c = 0; c < 4; c++) {
                    float wx = center[0] + right[0] * px[c] + up[0] * py[c];
                    float wy = center[1] + right[1] * px[c] + up[1] * py[c];
                    float wz = center[2] + right[2] * px[c] + up[2] * py[c];
                    if (wx < -0.5f || wx > 0.5f ||
                        wy < -0.5f || wy > 0.5f ||
                        wz < -0.5f || wz > 0.5f) {
                        glColor4f(0.0f, 0.0f, 0.0f, 0.0f);
                    } else {
                        int ci = world_to_cell(wx);
                        int cj = world_to_cell(wy);
                        int ck = world_to_cell(wz);
                        int idx = IX3(ci, cj, ck);
                        set_smoke_color(h_dens[idx], h_u[idx], h_v[idx], h_w[idx], alpha_scale);
                    }
                    glVertex3f(wx, wy, wz);
                }
            }
        }
        glEnd();
    }

    if (draw_border) {
        glDisable(GL_BLEND);
        glLineWidth(1.5f);
        glColor3f(0.90f, 0.92f, 0.95f);
        glBegin(GL_LINE_LOOP);
        glVertex3f(center[0] + (-right[0] - up[0]) * 0.5f,
                   center[1] + (-right[1] - up[1]) * 0.5f,
                   center[2] + (-right[2] - up[2]) * 0.5f);
        glVertex3f(center[0] + ( right[0] - up[0]) * 0.5f,
                   center[1] + ( right[1] - up[1]) * 0.5f,
                   center[2] + ( right[2] - up[2]) * 0.5f);
        glVertex3f(center[0] + ( right[0] + up[0]) * 0.5f,
                   center[1] + ( right[1] + up[1]) * 0.5f,
                   center[2] + ( right[2] + up[2]) * 0.5f);
        glVertex3f(center[0] + (-right[0] + up[0]) * 0.5f,
                   center[1] + (-right[1] + up[1]) * 0.5f,
                   center[2] + (-right[2] + up[2]) * 0.5f);
        glEnd();
        glLineWidth(1.0f);
        glEnable(GL_BLEND);
    }
}

static void draw_volume_slices(void) {
    if (!show_volume) return;

    int slices = (N > 80) ? 44 : 36;
    float step = 1.15f / (float)(slices - 1);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    for (int s = slices - 1; s >= 0; s--) {
        float offset = -0.575f + step * (float)s;
        draw_oriented_density_plane(offset, 0.055f, 1, 0);
    }
    glDisable(GL_BLEND);
}

static void draw_slice_plane(void) {
    if (!show_slice) return;
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    draw_oriented_density_plane(0.0f, 0.26f, 1, 1);
    glDisable(GL_BLEND);
}

static void draw_vector_field_3d(void) {
    if (!show_vectors) return;
    int stride = N / 8;
    if (stride < 5) stride = 5;
    float spacing = (float)stride / (float)N;

    glLineWidth(2.0f);
    glBegin(GL_LINES);
    for (int k = stride / 2 + 1; k <= N; k += stride) {
        for (int j = stride / 2 + 1; j <= N; j += stride) {
            for (int i = stride / 2 + 1; i <= N; i += stride) {
                int idx = IX3(i, j, k);
                float vx = h_u[idx], vy = h_v[idx], vz = h_w[idx];
                float speed = sqrtf(vx * vx + vy * vy + vz * vz);
                if (speed < 0.0002f) continue;
                float mag = 1.0f - expf(-speed * 0.7f);
                float len = spacing * (0.20f + 0.85f * mag);
                float inv = 1.0f / speed;
                float x = cell_coord(i), y = cell_coord(j), z = cell_coord(k);
                float ex = x + vx * inv * len;
                float ey = y + vy * inv * len;
                float ez = z + vz * inv * len;
                glColor3f(0.10f + 0.90f * mag,
                          0.85f + 0.15f * mag,
                          1.00f - 0.65f * mag);
                glVertex3f(x, y, z);
                glVertex3f(ex, ey, ez);
            }
        }
    }
    glEnd();
    glLineWidth(1.0f);
}

static void display(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    apply_camera_3d();
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    draw_cube_bounds();
    draw_input_plane_guide();
    draw_vector_field_3d();

    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);
    draw_volume_slices();
    glEnable(GL_DEPTH_TEST);
    draw_slice_plane();
    glDepthMask(GL_TRUE);
    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    char line[256];
    snprintf(line, sizeof(line), "3D GPU N=%d  plane=%.2f%s%s%s  %.2f ms  FPS=%.1f%s",
             N, camera_slice_offset, show_volume ? " volume" : "",
             show_slice ? " plane" : "", auto_source ? " auto" : "", current_ms, current_fps,
             paused ? "  [PAUSED]" : "");
    draw_text(0.01f, 0.97f, line);
    snprintf(line, sizeof(line), "LMB volumetric brush  WASD/QE move  RMB look  [] brush depth  x plane view  b volume  o auto  ESC quit");
    draw_text(0.01f, 0.02f, line);

    glutSwapBuffers();
}

static void idle(void) {
    update_camera();
    if (!paused) {
        get_from_ui();
        size_t bytes = volume_count() * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_dens_prev, h_dens_prev, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u_prev, h_u_prev, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_prev, h_v_prev, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w_prev, h_w_prev, bytes, cudaMemcpyHostToDevice));

        double t0 = now_seconds();
        vel_step3d(N, d_u, d_v, d_w, d_u_prev, d_v_prev, d_w_prev, visc, dt);
        dens_step3d(N, d_dens, d_dens_prev, d_u, d_v, d_w, diff, dt);
        fade_fields3d(N, d_dens, d_u, d_v, d_w, dissipation, 0.992f);
        CUDA_CHECK(cudaDeviceSynchronize());
        double t1 = now_seconds();

        accum_time += t1 - t0;
        frame_count++;
        CUDA_CHECK(cudaMemcpy(h_dens, d_dens, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_u, d_u, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_w, d_w, bytes, cudaMemcpyDeviceToHost));

        double now = now_seconds();
        if (now - last_fps_time > 0.5) {
            current_ms = (accum_time / frame_count) * 1000.0;
            current_fps = frame_count / (accum_time > 0.0 ? accum_time : 1.0);
            printf("3D N=%d plane=%.3f GPU=%.3f ms fps=%.1f cam=(%.2f %.2f %.2f)\n",
                   N, camera_slice_offset, current_ms, current_fps, cam_x, cam_y, cam_z);
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
    omx = mx = x;
    omy = my = y;
    if (button == GLUT_LEFT_BUTTON) mouse_down[0] = (state == GLUT_DOWN);
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
    keys[key] = 1;
    switch (key) {
        case 27: free_data(); exit(0);
        case 'p': case 'P': paused = !paused; break;
        case 'c': case 'C': clear_data(); break;
        case 'v': case 'V': show_vectors = !show_vectors; break;
        case 'x': case 'X': show_slice = !show_slice; break;
        case 'b': case 'B': show_volume = !show_volume; break;
        case 'o': case 'O': auto_source = !auto_source; break;
        case '[':
            camera_slice_offset -= 1.0f / (float)N;
            if (camera_slice_offset < -0.5f) camera_slice_offset = -0.5f;
            break;
        case ']':
            camera_slice_offset += 1.0f / (float)N;
            if (camera_slice_offset > 0.5f) camera_slice_offset = 0.5f;
            break;
        case 'f': case 'F':
            cam_x = 0.0f; cam_y = 0.0f; cam_z = 2.2f; cam_yaw = 0.0f; cam_pitch = 0.0f;
            camera_slice_offset = 0.0f;
            break;
    }
}

static void keyboard_up(unsigned char key, int x, int y) {
    (void)x; (void)y;
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
    camera_slice_offset = 0.0f;
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    printf("=== Stable Fluids 3D (GPU/CUDA slice viewer) ===\n");
    prompt_grid_size();
    printf("Grid: %d^3 interior cells, %.1f MB per scalar field\n",
           N, (double)(volume_count() * sizeof(float)) / (1024.0 * 1024.0));
    printf("Controls: LMB volumetric brush, WASD/QE move, RMB look, [] brush depth, x plane view, b volume, o auto source, v vectors, p pause, c clear, ESC quit\n\n");

    if (!allocate_data()) return 1;
    clear_data();

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(win_x, win_y);
    glutCreateWindow("Stable Fluids 3D - GPU/CUDA Volume Viewer");
    glClearColor(0, 0, 0, 1);

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);
    glutKeyboardUpFunc(keyboard_up);
    last_fps_time = now_seconds();

    glutMainLoop();
    return 0;
}
