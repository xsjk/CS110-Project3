#include "d2q9_bgk.h"
#include <immintrin.h>

/* The main processes in one step */
int collision(const t_param params, t_speed* cells, float* obstacles, int n_iter);
int streaming(const t_param params, t_speed* cells, int n_iter);
int boundary(const t_param params, t_speed* cells, float* inlets, int n_iter);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** collision(), obstacle(), streaming() & boundary()
*/
int timestep(const t_param params, t_speed* cells, float* inlets, float* obstacles, int n_iter)
{
  /* The main time overhead, you should mainly optimize these processes. */
  collision(params, cells, obstacles, n_iter);
  streaming(params, cells, n_iter);
  boundary(params, cells, inlets, n_iter);
  return EXIT_SUCCESS;
}

/*
** The collision of fluids in the cell is calculated using 
** the local equilibrium distribution and relaxation process
*/
int collision(const t_param params, t_speed* cells, float* obstacles, int n_iter) {
  /* loop over the cells in the grid
  ** the collision step is called before
  ** the streaming step and so values of interest
  ** are in the scratch-space grid */
  #pragma omp parallel for
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii+=8)
    {
      __m256 mask = _mm256_loadu_ps(&obstacles[ii + jj*params.nx]); // load our mask
      __m256 omega = _mm256_set1_ps(params.omega);

      float *speeds[NSPEEDS] = {
        &cells->speeds[0][ii + jj*params.nx],
        &cells->speeds[1][ii + jj*params.nx],
        &cells->speeds[2][ii + (params.ny - 1 - jj) * params.nx],
        &cells->speeds[3][ii + jj*params.nx],
        &cells->speeds[4][ii + jj*params.nx],
        &cells->speeds[5][(params.ny + ii - jj) + (params.ny - 1 - jj)*(params.nx + params.ny)],
        &cells->speeds[6][(ii + jj) + (params.ny - 1 - jj)*(params.nx + params.ny)],
        &cells->speeds[7][(params.ny + ii - jj) + jj*(params.nx + params.ny)],
        &cells->speeds[8][(ii + jj) + jj*(params.nx + params.ny)]
      };

      __m256 s[NSPEEDS] = {
        _mm256_loadu_ps(speeds[0]),
        _mm256_loadu_ps(speeds[1]),
        _mm256_loadu_ps(speeds[2]),
        _mm256_loadu_ps(speeds[3]),
        _mm256_loadu_ps(speeds[4]),
        _mm256_loadu_ps(speeds[5]),
        _mm256_loadu_ps(speeds[6]),
        _mm256_loadu_ps(speeds[7]),
        _mm256_loadu_ps(speeds[8])
      };
      
      /* compute local density total */
      __m256 local_density = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7] + s[8];
      __m256 _local_density = 3 / local_density;
      __m256 u[NSPEEDS];
      u[1] = ((s[1] + s[5] + s[8]) - (s[3] + s[6] + s[7])) * _local_density;
      u[2] = ((s[2] + s[5] + s[6]) - (s[4] + s[7] + s[8])) * _local_density;
      u[3] = -u[1];
      u[4] = -u[2];
      u[5] = u[1] + u[2];
      u[6] = u[2] - u[1];
      u[7] = -u[5];
      u[8] = -u[6];

      __m256 half = _mm256_set1_ps(.5);
      __m256 usq[4] = {
        u[1] * u[1] * half,
        u[2] * u[2] * half,
        u[5] * u[5] * half,
        u[6] * u[6] * half,
      };
      __m256 _sq = _mm256_fmadd_ps(usq[0] + usq[1], _mm256_set1_ps(-1.f/3.f), _mm256_set1_ps(1.f));
      
      __m256 w0 = 4.f / 9.f * local_density;
      __m256 w1 = 1.f / 9.f * local_density;
      __m256 w2 = 1.f / 36.f * local_density;

      /* relaxation step */
      _mm256_store_ps(speeds[0], _mm256_blendv_ps(_mm256_fmadd_ps(omega, _mm256_fmsub_ps(w0, _sq, s[0]), s[0]), s[0], mask));
      _mm256_store_ps(speeds[1], _mm256_blendv_ps(_mm256_fmadd_ps(omega, _mm256_fmsub_ps(w1, (_sq + u[1] + usq[0]), s[1]), s[1]), s[3], mask));
      _mm256_store_ps(speeds[2], _mm256_blendv_ps(_mm256_fmadd_ps(omega, _mm256_fmsub_ps(w1, (_sq + u[2] + usq[1]), s[2]), s[2]), s[4], mask));
      _mm256_store_ps(speeds[3], _mm256_blendv_ps(_mm256_fmadd_ps(omega, _mm256_fmsub_ps(w1, (_sq + u[3] + usq[0]), s[3]), s[3]), s[1], mask));
      _mm256_store_ps(speeds[4], _mm256_blendv_ps(_mm256_fmadd_ps(omega, _mm256_fmsub_ps(w1, (_sq + u[4] + usq[1]), s[4]), s[4]), s[2], mask));
      _mm256_storeu_ps(speeds[5], _mm256_blendv_ps(_mm256_fmadd_ps(omega, _mm256_fmsub_ps(w2, (_sq + u[5] + usq[2]), s[5]), s[5]), s[7], mask));
      _mm256_storeu_ps(speeds[6], _mm256_blendv_ps(_mm256_fmadd_ps(omega, _mm256_fmsub_ps(w2, (_sq + u[6] + usq[3]), s[6]), s[6]), s[8], mask));
      _mm256_storeu_ps(speeds[7], _mm256_blendv_ps(_mm256_fmadd_ps(omega, _mm256_fmsub_ps(w2, (_sq + u[7] + usq[2]), s[7]), s[7]), s[5], mask));
      _mm256_storeu_ps(speeds[8], _mm256_blendv_ps(_mm256_fmadd_ps(omega, _mm256_fmsub_ps(w2, (_sq + u[8] + usq[3]), s[8]), s[8]), s[6], mask));
      
    }
  }
  return EXIT_SUCCESS;
}

/*
** Particles flow to the corresponding cell according to their speed direaction.
*/
int streaming(const t_param params, t_speed* cells, int n_iter) {

  cells->speeds[2] += params.nx;
  cells->speeds[4] += params.nx;
  cells->speeds[5] += params.nx + params.ny;
  cells->speeds[6] += params.nx + params.ny;
  cells->speeds[7] += params.nx + params.ny;
  cells->speeds[8] += params.nx + params.ny;

  #pragma omp parallel for
  for (int jj = 0; jj < params.ny; jj++)
    for (int ii = 1; ii < params.nx; ii++) {
      cells->speeds[3][(ii-1) + jj*params.nx] = cells->speeds[3][ii + jj*params.nx]; /* west */
      cells->speeds[1][(params.nx-1 - (ii-1)) + jj*params.nx] = cells->speeds[1][(params.nx-1 - ii) + jj*params.nx]; /* east */
    }

  return EXIT_SUCCESS;
}

/*
** Work with boundary conditions. The upper and lower boundaries use the rebound plane, 
** the left border is the inlet of fixed speed, and 
** the right border is the open outlet of the first-order approximation.
*/
int boundary(const t_param params, t_speed* cells, float* inlets, int n_iter) {
  /* Set the constant coefficient */
  const float cst1 = 2.0/3.0;
  const float cst2 = 1.0/6.0;
  const float cst3 = 1.0/2.0;

  int ii, jj; 
  float local_density;
  
  for(ii = 0; ii < params.nx; ii++){
    // top wall (bounce)
    jj = params.ny -1;
    cells->speeds[4][ii + jj*params.nx] = cells->speeds[2][ii + (params.ny - 1 - jj - 1)*params.nx];
    cells->speeds[7][(params.ny + ii - jj) + jj*(params.nx + params.ny)] = cells->speeds[5][(params.ny + ii - jj) + (params.ny - 1 - jj - 1)*(params.nx + params.ny)];
    cells->speeds[8][(ii + jj) + jj*(params.nx + params.ny)] = cells->speeds[6][(ii + jj) + (params.ny - 1 - jj - 1)*(params.nx + params.ny)];
    // bottom wall (bounce)
    jj = 0;
    cells->speeds[2][ii + (params.ny - 1 - jj) * params.nx] = cells->speeds[4][ii + (jj - 1)*params.nx];
    cells->speeds[5][(params.ny + ii - jj) + (params.ny - 1 - jj)*(params.nx + params.ny)] = cells->speeds[7][(params.ny + ii - jj) + (jj - 1)*(params.nx + params.ny)];
    cells->speeds[6][(ii + jj) + (params.ny - 1 - jj)*(params.nx + params.ny)] = cells->speeds[8][(ii + jj) + (jj - 1)*(params.nx + params.ny)];
  }

  // left wall (inlet)
  ii = 0;
  for(jj = 0; jj < params.ny; jj++){
    local_density = ( cells->speeds[0][ii + jj*params.nx]
                      + cells->speeds[2][ii + (params.ny - 1 - jj) * params.nx]
                      + cells->speeds[4][ii + jj*params.nx]
                      + 2.0 * cells->speeds[3][ii + jj*params.nx]
                      + 2.0 * cells->speeds[6][(ii + jj) + (params.ny - 1 - jj)*(params.nx + params.ny)]
                      + 2.0 * cells->speeds[7][(params.ny + ii - jj) + jj*(params.nx + params.ny)]
                      )/(1.0 - inlets[jj]);

    cells->speeds[1][ii + jj*params.nx] = cells->speeds[3][ii + jj*params.nx]
                                        + cst1*local_density*inlets[jj];

    cells->speeds[5][(params.ny + ii - jj) + (params.ny - 1 - jj)*(params.nx + params.ny)] = cells->speeds[7][(params.ny + ii - jj) + jj*(params.nx + params.ny)]
                                        - cst3*(cells->speeds[2][ii + (params.ny - 1 - jj) * params.nx]-cells->speeds[4][ii + jj*params.nx])
                                        + cst2*local_density*inlets[jj];

    cells->speeds[8][(ii + jj) + jj*(params.nx + params.ny)] = cells->speeds[6][(ii + jj) + (params.ny - 1 - jj)*(params.nx + params.ny)]
                                        + cst3*(cells->speeds[2][ii + (params.ny - 1 - jj) * params.nx]-cells->speeds[4][ii + jj*params.nx])
                                        + cst2*local_density*inlets[jj];
  
  }

  // right wall (outlet)
  ii = params.nx-1;
  for(jj = 0; jj < params.ny; jj++){
    cells->speeds[0][ii + jj*params.nx] = cells->speeds[0][ii-1 + jj*params.nx];
    cells->speeds[1][ii + jj*params.nx] = cells->speeds[1][ii-1 + jj*params.nx];
    cells->speeds[2][ii + (params.ny - 1 - jj) * params.nx] = cells->speeds[2][ii-1 + (params.ny - 1 - jj) * params.nx];
    cells->speeds[3][ii + jj*params.nx] = cells->speeds[3][ii-1 + jj*params.nx];
    cells->speeds[4][ii + jj*params.nx] = cells->speeds[4][ii-1 + jj*params.nx];
    cells->speeds[5][(params.ny + ii - jj) + (params.ny - 1 - jj)*(params.nx + params.ny)] = cells->speeds[5][(params.ny + ii-1 - jj) + (params.ny - 1 - jj)*(params.nx + params.ny)];
    cells->speeds[6][(ii + jj) + (params.ny - 1 - jj)*(params.nx + params.ny)] = cells->speeds[6][(ii-1 + jj) + (params.ny - 1 - jj)*(params.nx + params.ny)];
    cells->speeds[7][(params.ny + ii - jj) + jj*(params.nx + params.ny)] = cells->speeds[7][(params.ny + ii - 1 - jj) + jj*(params.nx + params.ny)];
    cells->speeds[8][(ii + jj) + jj*(params.nx + params.ny)] = cells->speeds[8][(ii-1 + jj) + jj*(params.nx + params.ny)];
  }
  return EXIT_SUCCESS;
}
