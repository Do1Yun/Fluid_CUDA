// 2D Stable Fluids - GPU-only GLUT viewer
// Based on Jos Stam, "Real-Time Fluid Dynamics for Games" (GDC 2003)

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L  // for clock_gettime, CLOCK_MONOTONIC
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
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
static int brush_cells_divisor = 64;
static float velocity_vis_scale = 0.25f;
static int velocity_arrow_divisor = 32;

static float *h_u, *h_v, *h_u_prev, *h_v_prev;
static float *h_dens, *h_dens_prev;
static unsigned char *h_density_pixels;

static float *d_u, *d_v, *d_u_prev, *d_v_prev;
static float *d_dens, *d_dens_prev;
static GLuint density_tex = 0;

static int win_x = 512, win_y = 512;
static int mouse_down[3];
static int omx, omy, mx, my;
static int paused = 0;
static int draw_velocity = 0;

static int   frame_count   = 0;
static double fps_accum    = 0.0;
static double step_accum   = 0.0;
static double last_fps_time = 0.0;
static double current_fps  = 0.0;
static double current_step_ms = 0.0;

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

static void clear_data(void) {
    int size = (N + 2) * (N + 2);
    int bytes = size * sizeof(float);
    for (int i = 0; i < size; i++) {
        h_u[i] = h_v[i] = h_u_prev[i] = h_v_prev[i] = h_dens[i] = h_dens_prev[i] = 0.0f;
    }
        
    CUDA_CHECK(cudaMemcpy(d_u, h_u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_prev, h_u_prev, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_prev, h_v_prev, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dens, h_dens, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dens_prev, h_dens_prev, bytes, cudaMemcpyHostToDevice));
}

static int allocate_data(void) {
    int size = (N + 2) * (N + 2);
    int bytes = size * sizeof(float);
    
    h_u         = (float *)malloc(bytes);
    h_v         = (float *)malloc(bytes);
    h_u_prev    = (float *)malloc(bytes);
    h_v_prev    = (float *)malloc(bytes);
    h_dens      = (float *)malloc(bytes);
    h_dens_prev = (float *)malloc(bytes);
    h_density_pixels = (unsigned char *)malloc((size_t)N * N * 4);
    if (!h_u || !h_v || !h_u_prev || !h_v_prev || !h_dens || !h_dens_prev ||
        !h_density_pixels) {
        fprintf(stderr, "ERROR: out of memory\n");
        return 0;
    }

    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_u_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_v_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_dens, bytes));
    CUDA_CHECK(cudaMalloc(&d_dens_prev, bytes));

    init_solver(N);
    return 1;
}

static void free_data(void) {
    free(h_u); free(h_v); free(h_u_prev); free(h_v_prev);
    free(h_dens); free(h_dens_prev);
    free(h_density_pixels);
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
    
    free_solver();
}

static void get_from_UI(float *d, float *u_, float *v_) {
    int size = (N + 2) * (N + 2);
    for (int i = 0; i < size; i++)
        u_[i] = v_[i] = d[i] = 0.0f;

    if (!mouse_down[0] && !mouse_down[2]) return;

    int view_w = win_x;
    if (view_w <= 0) view_w = win_x;
    int local_mx = mx;
    int local_omx = omx;
    if (local_mx < 0) local_mx = 0;
    if (local_mx >= view_w) local_mx = view_w - 1;
    if (local_omx < 0) local_omx = 0;
    if (local_omx >= view_w) local_omx = view_w - 1;

    int i = (int)((local_mx / (float)view_w) * N + 1);
    int j = (int)(((win_y - my) / (float)win_y) * N + 1);
    if (i < 1 || i > N || j < 1 || j > N) return;

    int radius = N / brush_cells_divisor;
    if (radius < 2) radius = 2;
    float inv_radius2 = 1.0f / (float)(radius * radius);
    float du = force * (local_mx - local_omx);
    float dv = force * (omy - my);

    for (int jj = j - radius; jj <= j + radius; jj++) {
        if (jj < 1 || jj > N) continue;
        for (int ii = i - radius; ii <= i + radius; ii++) {
            if (ii < 1 || ii > N) continue;
            int dx = ii - i;
            int dy = jj - j;
            float dist2 = (float)(dx * dx + dy * dy);
            if (dist2 > (float)(radius * radius)) continue;

            float falloff = 1.0f - dist2 * inv_radius2;
            int idx = IX(ii, jj);
            if (mouse_down[0]) {
                u_[idx] += du * falloff;
                v_[idx] += dv * falloff;
            }
            if (mouse_down[2]) {
                d[idx] += source * falloff;
            }
        }
    }
    omx = mx;
    omy = my;
}

static void add_auto_smoke(float *d, float *v_) {
    int cx = (N + 2) / 2;
    int cy = (N + 2) / 2;
    int radius = N / brush_cells_divisor;
    if (radius < 2) radius = 2;
    float inv_radius2 = 1.0f / (float)(radius * radius);

    for (int jj = cy - radius; jj <= cy + radius; jj++) {
        if (jj < 1 || jj > N) continue;
        for (int ii = cx - radius; ii <= cx + radius; ii++) {
            if (ii < 1 || ii > N) continue;
            int dx = ii - cx;
            int dy = jj - cy;
            float dist2 = (float)(dx * dx + dy * dy);
            if (dist2 > (float)(radius * radius)) continue;

            float falloff = 1.0f - dist2 * inv_radius2;
            int idx = IX(ii, jj);
            d[idx] += source * falloff;
            v_[idx] += -2.0f * falloff;
        }
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

static void update_density_texture(const float *dens) {
    ensure_density_texture();

    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            float d = dens[IX(i, j)];
            if (d < 0.0f) d = 0.0f;
            if (d > 1.0f) d = 1.0f;
            unsigned char c = (unsigned char)(d * 255.0f);
            int out = ((j - 1) * N + (i - 1)) * 4;
            h_density_pixels[out + 0] = c;
            h_density_pixels[out + 1] = c;
            h_density_pixels[out + 2] = c;
            h_density_pixels[out + 3] = 255;
        }
    }

    glBindTexture(GL_TEXTURE_2D, density_tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, N, N,
                    GL_RGBA, GL_UNSIGNED_BYTE, h_density_pixels);
}

static void draw_density_field(const float *dens) {
    update_density_texture(dens);

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

    glLineWidth(1.5f);
    glBegin(GL_LINES);
    for (int i = stride / 2 + 1; i <= N; i += stride) {
        float x = (i - 0.5f) * h;
        for (int j = stride / 2 + 1; j <= N; j += stride) {
            float y = (j - 0.5f) * h;
            float vx = u[IX(i, j)];
            float vy = v[IX(i, j)];
            float speed = sqrtf(vx * vx + vy * vy);
            if (speed < 0.01f) continue;

            float magnitude = speed * velocity_vis_scale;
            if (magnitude > 0.85f) magnitude = 0.85f;
            float len = spacing * magnitude;
            float dir_x = vx / speed;
            float dir_y = vy / speed;
            float end_x = x + dir_x * len;
            float end_y = y + dir_y * len;
            float head_len = len * 0.30f;
            float head_w = len * 0.18f;
            float px = -dir_y;
            float py = dir_x;
            float intensity = speed * 0.20f;
            if (intensity > 1.0f) intensity = 1.0f;

            glColor3f(0.15f + 0.85f * intensity, 1.0f, 0.15f);
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

static void draw_panel(int x, int w, const char *name,
                       const float *dens, const float *u, const float *v,
                       double step_ms) {
    glViewport(x, 0, w, win_y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    draw_density_field(dens);
    if (draw_velocity) draw_velocity_field(u, v);

    char buf[160];
    snprintf(buf, sizeof(buf), "%s  %.2f ms%s", name, step_ms,
             paused ? "  [PAUSED]" : "");
    draw_text(0.01f, 0.93f, buf);
}

static void display(void) {
    glClear(GL_COLOR_BUFFER_BIT);

    draw_panel(0, win_x, "GPU", h_dens, h_u, h_v, current_step_ms);

    char buf[128];
    snprintf(buf, sizeof(buf), "N=%d  FPS=%.1f  v: velocity", N, current_fps);
    glViewport(0, 0, win_x, win_y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    draw_text(0.01f, 0.97f, buf);

    glutSwapBuffers();
}

static void idle(void) {
    if (!paused) {
        get_from_UI(h_dens_prev, h_u_prev, h_v_prev);
        add_auto_smoke(h_dens_prev, h_v_prev);

        int bytes = (N + 2) * (N + 2) * sizeof(float);

        CUDA_CHECK(cudaMemcpy(d_dens_prev, h_dens_prev, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u_prev, h_u_prev, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_prev, h_v_prev, bytes, cudaMemcpyHostToDevice));
        
        double step_t0 = now_seconds();
        vel_step(N, d_u, d_v, d_u_prev, d_v_prev, visc, dt);
        dens_step(N, d_dens, d_dens_prev, d_u, d_v, diff, dt);
        
        fade_fields(N, d_dens, d_u, d_v, dissipation, 0.99f);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        double step_t1 = now_seconds();

        step_accum += (step_t1 - step_t0);
        fps_accum += (step_t1 - step_t0);
        frame_count++;

        CUDA_CHECK(cudaMemcpy(h_dens, d_dens, bytes, cudaMemcpyDeviceToHost));
        if (draw_velocity) {
            CUDA_CHECK(cudaMemcpy(h_u, d_u, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost));
        }

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
    }
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);

    printf("=== Stable Fluids 2D (GPU/CUDA) ===\n");
    printf("Grid: N=%d (interior), total %dx%d cells\n",
           N, N + 2, N + 2);
    printf("Controls:\n");
    printf("  Left mouse drag  : add velocity\n");
    printf("  Right mouse drag : add density (paint)\n");
    printf("  c                : clear\n");
    printf("  p                : pause\n");
    printf("  v                : toggle velocity field\n");
    printf("  q / ESC          : quit\n\n");

    if (!allocate_data()) return 1;
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
