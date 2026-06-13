#ifndef SOLVER_H
#define SOLVER_H

// 2D Stable Fluids solver (CUDA version)
// Based on Jos Stam, "Real-Time Fluid Dynamics for Games" (GDC 2003)

void init_solver(int N);
void free_solver();

typedef struct SolverProfile {
    double source_add_ms;
    double diffuse_ms;
    double project_ms;
    double advect_ms;
    double boundary_ms;
    double obstacle_ms;
    double fade_ms;
} SolverProfile;

void solver_set_profile(SolverProfile *profile);

void dens_step(int N, float *d_x, float *d_x0, float *d_u, float *d_v,
               float diff, float dt, unsigned char *d_solid);
void vel_step(int N, float *d_u, float *d_v, float *d_u0, float *d_v0,
              float visc, float dt, unsigned char *d_solid);

void fade_fields(int N, float *d_dens, float *d_u, float *d_v,
                 float dissipation, float vel_damping, unsigned char *d_solid);

#endif
