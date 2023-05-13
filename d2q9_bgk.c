#include "d2q9_bgk.h"
#include <immintrin.h>

#define swap(a, b) {float tmp = a; a = b; b = tmp;}

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
  const __m256 c_sq = _mm256_set1_ps(1.f / 3.f); /* square of speed of sound */
  const __m256 w0 = _mm256_set1_ps(4.f / 9.f);   /* weighting factor */
  const __m256 w1 = _mm256_set1_ps(1.f / 9.f);   /* weighting factor */
  const __m256 w2 = _mm256_set1_ps(1.f / 36.f);  /* weighting factor */

  /* loop over the cells in the grid
  ** the collision step is called before
  ** the streaming step and so values of interest
  ** are in the scratch-space grid */
  // #pragma omp parallel for
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii+=8)
    {
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
      
      const __m256 mask = _mm256_loadu_ps(&obstacles[ii + jj*params.nx]); // load our mask
      __m256 temp1[NSPEEDS], temp2[NSPEEDS];
        /* compute local density total */
        __m256 local_density = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7] + s[8];
        __m256 u_x = ((s[1] + s[5] + s[8]) - (s[3] + s[6] + s[7])) / local_density;
        __m256 u_y = ((s[2] + s[5] + s[6]) - (s[4] + s[7] + s[8])) / local_density;

        /* velocity squared */
        __m256 u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        __m256 u[NSPEEDS] = {
          _mm256_setzero_ps(),
          u_x,
          u_y,
          -u_x,
          -u_y,
          u_x + u_y,
          -u_x + u_y,
          -u_x - u_y,
          u_x - u_y
        };

        /* equilibrium densities */
        __m256 d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */

        d_equ[0] = w0 * local_density * (1.f + u[0] / c_sq
                                         + (u[0] * u[0]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* relaxation step */
        temp1[0] = s[0] + params.omega * (d_equ[0] - s[0]);
        temp1[1] = s[1] + params.omega * (d_equ[1] - s[1]);
        temp1[2] = s[2] + params.omega * (d_equ[2] - s[2]);
        temp1[3] = s[3] + params.omega * (d_equ[3] - s[3]);
        temp1[4] = s[4] + params.omega * (d_equ[4] - s[4]);
        temp1[5] = s[5] + params.omega * (d_equ[5] - s[5]);
        temp1[6] = s[6] + params.omega * (d_equ[6] - s[6]);
        temp1[7] = s[7] + params.omega * (d_equ[7] - s[7]);
        temp1[8] = s[8] + params.omega * (d_equ[8] - s[8]);

        temp2[0] = _mm256_loadu_ps(speeds[0]);
        temp2[1] = _mm256_loadu_ps(speeds[3]);
        temp2[2] = _mm256_loadu_ps(speeds[4]);
        temp2[3] = _mm256_loadu_ps(speeds[1]);
        temp2[4] = _mm256_loadu_ps(speeds[2]);
        temp2[5] = _mm256_loadu_ps(speeds[7]);
        temp2[6] = _mm256_loadu_ps(speeds[8]);
        temp2[7] = _mm256_loadu_ps(speeds[5]);
        temp2[8] = _mm256_loadu_ps(speeds[6]);
      
        _mm256_storeu_ps(speeds[0], _mm256_blendv_ps(temp1[0], temp2[0], mask));
        _mm256_storeu_ps(speeds[1], _mm256_blendv_ps(temp1[1], temp2[1], mask));
        _mm256_storeu_ps(speeds[2], _mm256_blendv_ps(temp1[2], temp2[2], mask));
        _mm256_storeu_ps(speeds[3], _mm256_blendv_ps(temp1[3], temp2[3], mask));
        _mm256_storeu_ps(speeds[4], _mm256_blendv_ps(temp1[4], temp2[4], mask));
        _mm256_storeu_ps(speeds[5], _mm256_blendv_ps(temp1[5], temp2[5], mask));
        _mm256_storeu_ps(speeds[6], _mm256_blendv_ps(temp1[6], temp2[6], mask));
        _mm256_storeu_ps(speeds[7], _mm256_blendv_ps(temp1[7], temp2[7], mask));
        _mm256_storeu_ps(speeds[8], _mm256_blendv_ps(temp1[8], temp2[8], mask));
      
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
  
  // top wall (bounce)
  jj = params.ny -1;
  for(ii = 0; ii < params.nx; ii++){
    cells->speeds[4][ii + jj*params.nx] = cells->speeds[2][ii + (params.ny - 1 - jj - 1)*params.nx];
    cells->speeds[7][(params.ny + ii - jj) + jj*(params.nx + params.ny)] = cells->speeds[5][(params.ny + ii - jj) + (params.ny - 1 - jj - 1)*(params.nx + params.ny)];
    cells->speeds[8][(ii + jj) + jj*(params.nx + params.ny)] = cells->speeds[6][(ii + jj) + (params.ny - 1 - jj - 1)*(params.nx + params.ny)];
  }

  // bottom wall (bounce)
  jj = 0;
  for(ii = 0; ii < params.nx; ii++){
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
