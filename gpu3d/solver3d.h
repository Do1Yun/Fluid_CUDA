#ifndef SOLVER3D_H
#define SOLVER3D_H

void init_solver3d(int n);
void free_solver3d();

void dens_step3d(int n, float *d_x, float *d_x0,
                 float *d_u, float *d_v, float *d_w,
                 float diff, float dt);
void vel_step3d(int n, float *d_u, float *d_v, float *d_w,
                float *d_u0, float *d_v0, float *d_w0,
                float visc, float dt);
void fade_fields3d(int n, float *d_dens,
                   float *d_u, float *d_v, float *d_w,
                   float dissipation, float vel_damping);
void apply_sphere_obstacle3d(int n, float *d_dens,
                             float *d_u, float *d_v, float *d_w,
                             float cx, float cy, float cz, float radius);

#endif
