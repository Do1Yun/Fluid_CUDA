// 2D Stable Fluids - CPU-only GLUT viewer
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
static float auto_smoke_velocity = -0.75f;
static int brush_cells_divisor = 64;
static float smoke_color_speed_scale = 0.55f;
static float velocity_vis_scale = 0.70f;
static int velocity_arrow_divisor = 24;

static float *u, *v, *u_prev, *v_prev;
static float *dens, *dens_prev;
static unsigned char *solid;

static int win_x = 512, win_y = 512;
static int mouse_down[3];
static int omx, omy, mx, my;
static int paused = 0;
static int draw_velocity = 1;
static int auto_smoke = 1;
static int obstacle_mode = 0;

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

typedef enum ProfileSection {
    PROFILE_SOURCE_ADD = 0,
    PROFILE_DIFFUSE,
    PROFILE_PROJECT,
    PROFILE_ADVECT,
    PROFILE_BOUNDARY,
    PROFILE_OBSTACLE,
    PROFILE_FADE,
    PROFILE_COUNT
} ProfileSection;

typedef struct BenchmarkProfile {
    double source_add_ms;
    double diffuse_ms;
    double project_ms;
    double advect_ms;
    double boundary_ms;
    double obstacle_ms;
    double fade_ms;
    int stack[16];
    double stack_start[16];
    int stack_depth;
} BenchmarkProfile;

static BenchmarkProfile *active_profile = NULL;

static void reset_profile(BenchmarkProfile *profile) {
    memset(profile, 0, sizeof(*profile));
}

static double *profile_bucket(BenchmarkProfile *profile, int section) {
    switch (section) {
    case PROFILE_SOURCE_ADD: return &profile->source_add_ms;
    case PROFILE_DIFFUSE: return &profile->diffuse_ms;
    case PROFILE_PROJECT: return &profile->project_ms;
    case PROFILE_ADVECT: return &profile->advect_ms;
    case PROFILE_BOUNDARY: return &profile->boundary_ms;
    case PROFILE_OBSTACLE: return &profile->obstacle_ms;
    case PROFILE_FADE: return &profile->fade_ms;
    default: return NULL;
    }
}

static void profile_begin(int section) {
    if (!active_profile || active_profile->stack_depth >= 16) return;
    double now = now_seconds();
    if (active_profile->stack_depth > 0) {
        int parent = active_profile->stack[active_profile->stack_depth - 1];
        double *bucket = profile_bucket(active_profile, parent);
        if (bucket) *bucket += (now - active_profile->stack_start[active_profile->stack_depth - 1]) * 1000.0;
    }
    active_profile->stack[active_profile->stack_depth] = section;
    active_profile->stack_start[active_profile->stack_depth] = now;
    active_profile->stack_depth++;
}

static void profile_end(int section) {
    if (!active_profile || active_profile->stack_depth <= 0) return;
    double now = now_seconds();
    int top = active_profile->stack_depth - 1;
    int current = active_profile->stack[top];
    double *bucket = profile_bucket(active_profile, current);
    if (bucket) *bucket += (now - active_profile->stack_start[top]) * 1000.0;
    active_profile->stack_depth--;
    if (active_profile->stack_depth > 0) {
        active_profile->stack_start[active_profile->stack_depth - 1] = now;
    }
    (void)section;
}

static void add_source_cpu(float *x, const float *s, float dt) {
    profile_begin(PROFILE_SOURCE_ADD);
    int size = (N + 2) * (N + 2);
    for (int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
    profile_end(PROFILE_SOURCE_ADD);
}

static void apply_solid_scalar_cpu(float *x) {
    if (!solid) return;
    profile_begin(PROFILE_BOUNDARY);
    for (int j = 0; j < N + 2; j++) {
        for (int i = 0; i < N + 2; i++) {
            if (solid[IX(i, j)]) {
                float sum = 0.0f;
                int count = 0;
                if (i > 1 && !solid[IX(i - 1, j)]) { sum += x[IX(i - 1, j)]; count++; }
                if (i < N && !solid[IX(i + 1, j)]) { sum += x[IX(i + 1, j)]; count++; }
                if (j > 1 && !solid[IX(i, j - 1)]) { sum += x[IX(i, j - 1)]; count++; }
                if (j < N && !solid[IX(i, j + 1)]) { sum += x[IX(i, j + 1)]; count++; }
                x[IX(i, j)] = (count > 0) ? sum / (float)count : 0.0f;
            }
        }
    }
    profile_end(PROFILE_BOUNDARY);
}

static void apply_solid_velocity_cpu(float *u_, float *v_) {
    if (!solid) return;
    profile_begin(PROFILE_BOUNDARY);
    for (int j = 0; j < N + 2; j++) {
        for (int i = 0; i < N + 2; i++) {
            int idx = IX(i, j);
            if (solid[idx]) {
                u_[idx] = 0.0f;
                v_[idx] = 0.0f;
                continue;
            }
            if (i > 1 && solid[IX(i - 1, j)] && u_[idx] < 0.0f) u_[idx] = 0.0f;
            if (i < N && solid[IX(i + 1, j)] && u_[idx] > 0.0f) u_[idx] = 0.0f;
            if (j > 1 && solid[IX(i, j - 1)] && v_[idx] < 0.0f) v_[idx] = 0.0f;
            if (j < N && solid[IX(i, j + 1)] && v_[idx] > 0.0f) v_[idx] = 0.0f;
        }
    }
    profile_end(PROFILE_BOUNDARY);
}

static void set_bnd_cpu(int b, float *x) {
    profile_begin(PROFILE_BOUNDARY);
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
    profile_end(PROFILE_BOUNDARY);
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
    profile_begin(PROFILE_DIFFUSE);
    float a = dt * diff * N * N;
    lin_solve_cpu(b, x, x0, a, 1.0f + 4.0f * a);
    profile_end(PROFILE_DIFFUSE);
}

static float sample_fluid_bilinear_cpu(const float *field,
                                       int i, int j,
                                       int i0, int i1, int j0, int j1,
                                       float s0, float s1, float t0, float t1) {
    float value = 0.0f;
    float weight = 0.0f;
    float current = field[IX(i, j)];

    int idx = IX(i0, j0);
    float w = s0 * t0;
    if (!solid || !solid[idx]) { value += w * field[idx]; weight += w; }
    idx = IX(i0, j1);
    w = s0 * t1;
    if (!solid || !solid[idx]) { value += w * field[idx]; weight += w; }
    idx = IX(i1, j0);
    w = s1 * t0;
    if (!solid || !solid[idx]) { value += w * field[idx]; weight += w; }
    idx = IX(i1, j1);
    w = s1 * t1;
    if (!solid || !solid[idx]) { value += w * field[idx]; weight += w; }

    return (weight > 0.000001f) ? value / weight : current;
}

static void advect_cpu(int b, float *d, const float *d0, const float *u, const float *v, float dt) {
    profile_begin(PROFILE_ADVECT);
    float dt0 = dt * N;
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            if (solid && solid[IX(i, j)]) {
                d[IX(i, j)] = d0[IX(i, j)];
                continue;
            }
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
            d[IX(i, j)] = sample_fluid_bilinear_cpu(d0, i, j,
                                                    i0, i1, j0, j1,
                                                    s0, s1, t0, t1);
        }
    }
    set_bnd_cpu(b, d);
    profile_end(PROFILE_ADVECT);
}

static void project_cpu(float *u, float *v, float *p, float *div) {
    profile_begin(PROFILE_PROJECT);
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            int idx = IX(i, j);
            if (solid && solid[idx]) {
                div[idx] = 0.0f;
                p[idx] = 0.0f;
                continue;
            }
            float u_r = (!solid || !solid[IX(i + 1, j)]) ? u[IX(i + 1, j)] : 0.0f;
            float u_l = (!solid || !solid[IX(i - 1, j)]) ? u[IX(i - 1, j)] : 0.0f;
            float v_t = (!solid || !solid[IX(i, j + 1)]) ? v[IX(i, j + 1)] : 0.0f;
            float v_b = (!solid || !solid[IX(i, j - 1)]) ? v[IX(i, j - 1)] : 0.0f;
            div[idx] = -0.5f * (u_r - u_l + v_t - v_b) / N;
            p[idx] = 0.0f;
        }
    }
    set_bnd_cpu(0, div);
    set_bnd_cpu(0, p);
    apply_solid_scalar_cpu(div);
    apply_solid_scalar_cpu(p);
    lin_solve_cpu(0, p, div, 1.0f, 4.0f);
    apply_solid_scalar_cpu(p);

    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            int idx = IX(i, j);
            if (solid && solid[idx]) {
                u[idx] = 0.0f;
                v[idx] = 0.0f;
                continue;
            }
            float p_c = p[idx];
            float p_r = (!solid || !solid[IX(i + 1, j)]) ? p[IX(i + 1, j)] : p_c;
            float p_l = (!solid || !solid[IX(i - 1, j)]) ? p[IX(i - 1, j)] : p_c;
            float p_t = (!solid || !solid[IX(i, j + 1)]) ? p[IX(i, j + 1)] : p_c;
            float p_b = (!solid || !solid[IX(i, j - 1)]) ? p[IX(i, j - 1)] : p_c;
            u[idx] -= 0.5f * N * (p_r - p_l);
            v[idx] -= 0.5f * N * (p_t - p_b);
        }
    }
    set_bnd_cpu(1, u);
    set_bnd_cpu(2, v);
    profile_end(PROFILE_PROJECT);
}

static void dens_step_cpu(float *x, float *x0, float *u, float *v, float diff, float dt) {
    add_source_cpu(x, x0, dt);
    apply_solid_scalar_cpu(x);
    SWAP(x0, x);
    diffuse_cpu(0, x, x0, diff, dt);
    apply_solid_scalar_cpu(x);
    SWAP(x0, x);
    advect_cpu(0, x, x0, u, v, dt);
    apply_solid_scalar_cpu(x);
}

static void vel_step_cpu(float *u, float *v, float *u0, float *v0, float visc, float dt) {
    add_source_cpu(u, u0, dt);
    add_source_cpu(v, v0, dt);
    apply_solid_velocity_cpu(u, v);
    SWAP(u0, u);
    diffuse_cpu(1, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse_cpu(2, v, v0, visc, dt);
    apply_solid_velocity_cpu(u, v);
    project_cpu(u, v, u0, v0);
    apply_solid_velocity_cpu(u, v);
    SWAP(u0, u);
    SWAP(v0, v);
    advect_cpu(1, u, u0, u0, v0, dt);
    advect_cpu(2, v, v0, u0, v0, dt);
    apply_solid_velocity_cpu(u, v);
    project_cpu(u, v, u0, v0);
    apply_solid_velocity_cpu(u, v);
}

static void fade_fields_cpu(float *dens, float *u, float *v, float dissipation, float vel_damping) {
    profile_begin(PROFILE_FADE);
    int size = (N + 2) * (N + 2);
    for (int i = 0; i < size; i++) {
        dens[i] *= dissipation;
        u[i] *= vel_damping;
        v[i] *= vel_damping;
    }
    apply_solid_scalar_cpu(dens);
    apply_solid_velocity_cpu(u, v);
    profile_end(PROFILE_FADE);
}

static void clear_data(void) {
    int size = (N + 2) * (N + 2);
    for (int i = 0; i < size; i++) {
        u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
    }
}

static void clear_obstacles(void) {
    int size = (N + 2) * (N + 2);
    memset(solid, 0, (size_t)size);
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
    solid     = (unsigned char *)malloc((size_t)size);
    
    if (!u || !v || !u_prev || !v_prev || !dens || !dens_prev || !solid) {
        fprintf(stderr, "ERROR: out of memory\n");
        return 0;
    }
    return 1;
}

static void free_data(void) {
    free(u); free(v); free(u_prev); free(v_prev);
    free(dens); free(dens_prev);
    free(solid);
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
            solid[IX(ii, jj)] = erase ? 0 : 1;
            u[IX(ii, jj)] = v[IX(ii, jj)] = dens[IX(ii, jj)] = 0.0f;
        }
    }
}

static void get_from_UI(float *d, float *u_, float *v_) {
    int size = (N + 2) * (N + 2);
    for (int i = 0; i < size; i++)
        u_[i] = v_[i] = d[i] = 0.0f;

    if (!mouse_down[0] && !mouse_down[2]) return;

    if (obstacle_mode) {
        if (mouse_down[0]) paint_obstacle_at(0);
        if (mouse_down[2]) paint_obstacle_at(1);
        return;
    }

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
            v_[idx] += auto_smoke_velocity * falloff;
        }
    }
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

static void set_density_velocity_color(float dens, float u_, float v_) {
    float visual_dens = 1.0f - expf(-dens * 0.18f);
    if (visual_dens < 0.0f) visual_dens = 0.0f;
    if (visual_dens > 1.0f) visual_dens = 1.0f;

    float speed = sqrtf(u_ * u_ + v_ * v_);
    float t = 1.0f - expf(-speed * smoke_color_speed_scale);
    if (t > 1.0f) t = 1.0f;

    float shade = visual_dens * (0.72f + 0.28f * t);
    float tint_r, tint_g, tint_b;
    hsv_to_rgb(0.58f - 0.48f * t, 0.78f, 1.0f, &tint_r, &tint_g, &tint_b);

    float tint_strength = 0.10f + 0.42f * t;
    float r = shade * ((1.0f - tint_strength) + tint_strength * tint_r);
    float g = shade * ((1.0f - tint_strength) + tint_strength * tint_g);
    float b = shade * ((1.0f - tint_strength) + tint_strength * tint_b);
    glColor3f(r, g, b);
}

static void set_cell_color(int i, int j, float dens_value, float u_value, float v_value) {
    if (solid && solid[IX(i, j)]) {
        glColor3f(0.10f, 0.12f, 0.14f);
    } else {
        set_density_velocity_color(dens_value, u_value, v_value);
    }
}

static void draw_density_field(const float *dens, const float *u_, const float *v_) {
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
            set_cell_color(i,     j,     d00, u_[IX(i,     j)],     v_[IX(i,     j)]);
            glVertex2f(x,     y);
            set_cell_color(i + 1, j,     d10, u_[IX(i + 1, j)],     v_[IX(i + 1, j)]);
            glVertex2f(x + h, y);
            set_cell_color(i + 1, j + 1, d11, u_[IX(i + 1, j + 1)], v_[IX(i + 1, j + 1)]);
            glVertex2f(x + h, y + h);
            set_cell_color(i,     j + 1, d01, u_[IX(i,     j + 1)], v_[IX(i,     j + 1)]);
            glVertex2f(x,     y + h);
        }
    }
    glEnd();
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

static void draw_panel(int x, int w, const char *name,
                       const float *dens, const float *u, const float *v,
                       double step_ms) {
    glViewport(x, 0, w, win_y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    draw_density_field(dens, u, v);
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
        get_from_UI(dens_prev, u_prev, v_prev);
        if (auto_smoke && !obstacle_mode) add_auto_smoke(dens_prev, v_prev);

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

static int clamp_cell_cpu(int value) {
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

    *center_i = clamp_cell_cpu((int)(x * (float)N) + 1);
    *center_j = clamp_cell_cpu((int)(y * (float)N) + 1);
    *density_value = source * ((pass == 0) ? 1.00f : 0.72f);
    *du = force * (0.90f * cosf(phase * 1.31f) + 0.25f * sinf(phase * 0.47f));
    *dv = force * (0.90f * sinf(phase * 1.07f) - 0.20f * cosf(phase * 0.73f));
}

static void add_benchmark_blob_cpu(float *d, float *u_, float *v_,
                                   int center_i, int center_j, int radius,
                                   float density_value, float du, float dv) {
    float inv_radius2 = 1.0f / (float)(radius * radius);
    for (int j = center_j - radius; j <= center_j + radius; j++) {
        if (j < 1 || j > N) continue;
        for (int i = center_i - radius; i <= center_i + radius; i++) {
            if (i < 1 || i > N) continue;
            int dx = i - center_i;
            int dy = j - center_j;
            float dist2 = (float)(dx * dx + dy * dy);
            if (dist2 > (float)(radius * radius)) continue;
            float falloff = 1.0f - dist2 * inv_radius2;
            int idx = IX(i, j);
            d[idx] += density_value * falloff;
            u_[idx] += du * falloff;
            v_[idx] += dv * falloff;
        }
    }
}

static void prepare_benchmark_sources_cpu(int frame) {
    int bytes = (N + 2) * (N + 2) * (int)sizeof(float);
    memset(dens_prev, 0, bytes);
    memset(u_prev, 0, bytes);
    memset(v_prev, 0, bytes);

    int radius = N / 32;
    if (radius < 3) radius = 3;
    for (int pass = 0; pass < 2; pass++) {
        int center_i, center_j;
        float density_value, du, dv;
        benchmark_source_params_2d(frame, pass, &center_i, &center_j,
                                   &density_value, &du, &dv);
        add_benchmark_blob_cpu(dens_prev, u_prev, v_prev,
                               center_i, center_j, radius,
                               density_value, du, dv);
    }
}

static void compute_benchmark_metrics_cpu(double *density_sum,
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

    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            int idx = IX(i, j);
            if (solid && solid[idx]) continue;
            float dens_value = dens[idx];
            float speed2 = u[idx] * u[idx] + v[idx] * v[idx];
            float div = 0.5f * (float)N *
                        (u[IX(i + 1, j)] - u[IX(i - 1, j)] +
                         v[IX(i, j + 1)] - v[IX(i, j - 1)]);
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

static void write_benchmark_csv_cpu(const char *mode,
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
    fprintf(fp, "%s,2d,%s,%s,%d,2,%lld,%.6f,%d,%d,%.6f,%.6f,%.6f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6e,%.6e,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
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

static int run_benchmark_cpu(const BenchmarkConfig *cfg) {
    N = cfg->grid_size;
    printf("=== Stable Fluids 2D (CPU benchmark) ===\n");
    printf("Benchmark input: deterministic_2d_v1, no GLUT, no user input\n");
    printf("Grid: N=%d, warmup=%d, frames=%d\n", N, cfg->warmup, cfg->frames);

    if (!allocate_data()) return 1;
    clear_obstacles();
    clear_data();

    int total_frames = cfg->warmup + cfg->frames;
    double source_seconds = 0.0;
    double step_seconds = 0.0;
    BenchmarkProfile profile_sum;
    reset_profile(&profile_sum);

    for (int frame = 0; frame < total_frames; frame++) {
        double source_t0 = now_seconds();
        prepare_benchmark_sources_cpu(frame);
        double source_t1 = now_seconds();

        BenchmarkProfile frame_profile;
        reset_profile(&frame_profile);
        active_profile = &frame_profile;
        double step_t0 = now_seconds();
        vel_step_cpu(u, v, u_prev, v_prev, visc, dt);
        dens_step_cpu(dens, dens_prev, u, v, diff, dt);
        fade_fields_cpu(dens, u, v, dissipation, 0.99f);
        double step_t1 = now_seconds();
        active_profile = NULL;

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
    compute_benchmark_metrics_cpu(&density_sum, &density_max,
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
    printf("benchmark_result,cpu2d,%d,%d,%d,%.6f,%.6f,%.6f,%.3f,%.6f,%.6f,%.6f,%.6e,%.6e,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
           N, cfg->warmup, cfg->frames,
           source_ms, step_ms, total_ms, fps,
           density_sum, density_max, velocity_l2,
           divergence_l2, divergence_max,
           source_add_ms, diffuse_ms, project_ms, advect_ms,
           boundary_ms, obstacle_ms, fade_ms, other_ms);
    if (cfg->save_csv) {
        write_benchmark_csv_cpu("cpu2d", "deterministic_2d_v1",
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
        return run_benchmark_cpu(&bench_cfg);
    }

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
