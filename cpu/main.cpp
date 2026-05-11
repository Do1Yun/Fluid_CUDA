// 2D Stable Fluids - CPU-only GLUT viewer
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

#ifdef __APPLE__
  #include <GLUT/glut.h>
#else
  #include <GL/glut.h>
#endif

#define SIZE 1024

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

static float *u, *v, *u_prev, *v_prev;
static float *dens, *dens_prev;

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

static void add_source_cpu(float *x, const float *s, float dt) {
    int size = (N + 2) * (N + 2);
    for (int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

static void set_bnd_cpu(int b, float *x) {
    for (int i = 1; i <= N; i++) {
        x[IX(0, i)]     = (b == 1) ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(N + 1, i)] = (b == 1) ? -x[IX(N, i)] : x[IX(N, i)];
        x[IX(i, 0)]     = (b == 2) ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, N + 1)] = (b == 2) ? -x[IX(i, N)] : x[IX(i, N)];
    }
    x[IX(0, 0)]         = 0.5f * (x[IX(1, 0)]     + x[IX(0, 1)]);
    x[IX(0, N + 1)]     = 0.5f * (x[IX(1, N + 1)] + x[IX(0, N)]);
    x[IX(N + 1, 0)]     = 0.5f * (x[IX(N, 0)]     + x[IX(N + 1, 1)]);
    x[IX(N + 1, N + 1)] = 0.5f * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

static void lin_solve_cpu(int b, float *x, const float *x0, float a, float c) {
    for (int k = 0; k < 20; k++) {
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                x[IX(i, j)] = (x0[IX(i, j)] +
                               a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                                    x[IX(i, j - 1)] + x[IX(i, j + 1)])) / c;
            }
        }
        set_bnd_cpu(b, x);
    }
}

static void diffuse_cpu(int b, float *x, const float *x0, float diff, float dt) {
    float a = dt * diff * N * N;
    lin_solve_cpu(b, x, x0, a, 1.0f + 4.0f * a);
}

static void advect_cpu(int b, float *d, const float *d0, const float *u, const float *v, float dt) {
    float dt0 = dt * N;
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
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
    set_bnd_cpu(b, d);
}

static void project_cpu(float *u, float *v, float *p, float *div) {
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] +
                                     v[IX(i, j + 1)] - v[IX(i, j - 1)]) / N;
            p[IX(i, j)] = 0.0f;
        }
    }
    set_bnd_cpu(0, div);
    set_bnd_cpu(0, p);
    lin_solve_cpu(0, p, div, 1.0f, 4.0f);

    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            u[IX(i, j)] -= 0.5f * N * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
            v[IX(i, j)] -= 0.5f * N * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
        }
    }
    set_bnd_cpu(1, u);
    set_bnd_cpu(2, v);
}

static void dens_step_cpu(float *x, float *x0, float *u, float *v, float diff, float dt) {
    add_source_cpu(x, x0, dt);
    SWAP(x0, x);
    diffuse_cpu(0, x, x0, diff, dt);
    SWAP(x0, x);
    advect_cpu(0, x, x0, u, v, dt);
}

static void vel_step_cpu(float *u, float *v, float *u0, float *v0, float visc, float dt) {
    add_source_cpu(u, u0, dt);
    add_source_cpu(v, v0, dt);
    SWAP(u0, u);
    diffuse_cpu(1, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse_cpu(2, v, v0, visc, dt);
    project_cpu(u, v, u0, v0);
    SWAP(u0, u);
    SWAP(v0, v);
    advect_cpu(1, u, u0, u0, v0, dt);
    advect_cpu(2, v, v0, u0, v0, dt);
    project_cpu(u, v, u0, v0);
}

static void fade_fields_cpu(float *dens, float *u, float *v, float dissipation, float vel_damping) {
    int size = (N + 2) * (N + 2);
    for (int i = 0; i < size; i++) {
        dens[i] *= dissipation;
        u[i] *= vel_damping;
        v[i] *= vel_damping;
    }
}

static void clear_data(void) {
    int size = (N + 2) * (N + 2);
    for (int i = 0; i < size; i++) {
        u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
    }
}

static int allocate_data(void) {
    int size = (N + 2) * (N + 2);
    int bytes = size * sizeof(float);
    
    u         = (float *)malloc(bytes);
    v         = (float *)malloc(bytes);
    u_prev    = (float *)malloc(bytes);
    v_prev    = (float *)malloc(bytes);
    dens      = (float *)malloc(bytes);
    dens_prev = (float *)malloc(bytes);
    
    if (!u || !v || !u_prev || !v_prev || !dens || !dens_prev) {
        fprintf(stderr, "ERROR: out of memory\n");
        return 0;
    }
    return 1;
}

static void free_data(void) {
    free(u); free(v); free(u_prev); free(v_prev);
    free(dens); free(dens_prev);
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

static void draw_density_field(const float *dens) {
    float h = 1.0f / N;
    glBegin(GL_QUADS);
    for (int i = 0; i <= N; i++) {
        float x = (i - 0.5f) * h;
        for (int j = 0; j <= N; j++) {
            float y = (j - 0.5f) * h;
            float d00 = dens[IX(i,     j)];
            float d01 = dens[IX(i,     j + 1)];
            float d10 = dens[IX(i + 1, j)];
            float d11 = dens[IX(i + 1, j + 1)];
            glColor3f(d00, d00, d00); glVertex2f(x,     y);
            glColor3f(d10, d10, d10); glVertex2f(x + h, y);
            glColor3f(d11, d11, d11); glVertex2f(x + h, y + h);
            glColor3f(d01, d01, d01); glVertex2f(x,     y + h);
        }
    }
    glEnd();
}

static void draw_velocity_field(const float *u, const float *v) {
    float h = 1.0f / N;
    int stride = N / velocity_arrow_divisor;
    if (stride < 4) stride = 4;
    float spacing = stride * h;

    glPointSize(3.0f);
    glColor3f(0.0f, 0.75f, 0.10f);
    glBegin(GL_POINTS);
    for (int i = stride / 2 + 1; i <= N; i += stride) {
        float x = (i - 0.5f) * h;
        for (int j = stride / 2 + 1; j <= N; j += stride) {
            float y = (j - 0.5f) * h;
            glVertex2f(x, y);
        }
    }
    glEnd();
    glPointSize(1.0f);

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

    draw_panel(0, win_x, "CPU", dens, u, v, current_step_ms);

    char buf[128];
    snprintf(buf, sizeof(buf), "N=%d  FPS=%.1f  v: velocity", N, current_fps);
    glViewport(0, 0, win_x, win_y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    draw_text(0.01f, 0.97f, buf);
    draw_text(0.01f, 0.02f,
              "LMB: velocity | RMB: smoke | c: clear | p: pause | v: vectors | q/ESC: quit");

    glutSwapBuffers();
}

static void idle(void) {
    if (!paused) {
        get_from_UI(dens_prev, u_prev, v_prev);
        add_auto_smoke(dens_prev, v_prev);

        double step_t0 = now_seconds();
        vel_step_cpu(u, v, u_prev, v_prev, visc, dt);
        dens_step_cpu(dens, dens_prev, u, v, diff, dt);
        fade_fields_cpu(dens, u, v, dissipation, 0.99f);
        double step_t1 = now_seconds();

        step_accum += (step_t1 - step_t0);
        fps_accum += (step_t1 - step_t0);
        frame_count++;

        double now = now_seconds();
        if (now - last_fps_time >= 0.5) {
            current_fps = frame_count / (fps_accum > 0 ? fps_accum : 1.0);
            current_step_ms = (step_accum / frame_count) * 1000.0;
            printf("N=%d  CPU=%.3f ms  fps=%.1f\n",
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

int main(int argc, char **argv) {
    glutInit(&argc, argv);

    printf("=== Stable Fluids 2D (CPU) ===\n");
    prompt_grid_size();
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
    glutCreateWindow("Stable Fluids 2D - CPU");

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
