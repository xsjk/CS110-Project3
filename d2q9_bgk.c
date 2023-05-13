#include "d2q9_bgk.h"


/* The main processes in one step */
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int streaming(const t_param params, t_speed* cells, t_speed* tmp_cells);
int obstacle(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int boundary(const t_param params, t_speed* cells, t_speed* tmp_cells, float* inlets);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** collision(), obstacle(), streaming() & boundary()
*/
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, float* inlets, int* obstacles)
{
  /* The main time overhead, you should mainly optimize these processes. */
  collision(params, cells, tmp_cells, obstacles);
  obstacle(params, cells, tmp_cells, obstacles);
  streaming(params, cells, tmp_cells);
  boundary(params, cells, tmp_cells, inlets);
  return EXIT_SUCCESS;
}

/*
** The collision of fluids in the cell is calculated using 
** the local equilibrium distribution and relaxation process
*/
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles) {
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */

  /* loop over the cells in the grid
  ** the collision step is called before
  ** the streaming step and so values of interest
  ** are in the scratch-space grid */
  #pragma omp parallel for
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      if (!obstacles[ii + jj*params.nx]){
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells->speeds[kk][ii + jj*params.nx];
        }

        /* compute x velocity component */
        float u_x = (cells->speeds[1][ii + jj*params.nx]
                      + cells->speeds[5][ii + jj*params.nx]
                      + cells->speeds[8][ii + jj*params.nx]
                      - (cells->speeds[3][ii + jj*params.nx]
                         + cells->speeds[6][ii + jj*params.nx]
                         + cells->speeds[7][ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells->speeds[2][ii + jj*params.nx]
                      + cells->speeds[5][ii + jj*params.nx]
                      + cells->speeds[6][ii + jj*params.nx]
                      - (cells->speeds[4][ii + jj*params.nx]
                         + cells->speeds[7][ii + jj*params.nx]
                         + cells->speeds[8][ii + jj*params.nx]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[0] = 0;            /* zero */
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
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
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          tmp_cells->speeds[kk][ii + jj*params.nx] = cells->speeds[kk][ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[kk] - cells->speeds[kk][ii + jj*params.nx]);
        }
      }
    }
  }
  return EXIT_SUCCESS;
}

/*
** For obstacles, mirror their speed.
*/
int obstacle(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles) {

  /* loop over the cells in the grid */
  #pragma omp parallel for
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after collision, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells->speeds[0][ii + jj*params.nx] = cells->speeds[0][ii + jj*params.nx];
        tmp_cells->speeds[1][ii + jj*params.nx] = cells->speeds[3][ii + jj*params.nx];
        tmp_cells->speeds[2][ii + jj*params.nx] = cells->speeds[4][ii + jj*params.nx];
        tmp_cells->speeds[3][ii + jj*params.nx] = cells->speeds[1][ii + jj*params.nx];
        tmp_cells->speeds[4][ii + jj*params.nx] = cells->speeds[2][ii + jj*params.nx];
        tmp_cells->speeds[5][ii + jj*params.nx] = cells->speeds[7][ii + jj*params.nx];
        tmp_cells->speeds[6][ii + jj*params.nx] = cells->speeds[8][ii + jj*params.nx];
        tmp_cells->speeds[7][ii + jj*params.nx] = cells->speeds[5][ii + jj*params.nx];
        tmp_cells->speeds[8][ii + jj*params.nx] = cells->speeds[6][ii + jj*params.nx];
      }
    }
  }
  return EXIT_SUCCESS;
}

/*
** Particles flow to the corresponding cell according to their speed direaction.
*/
int streaming(const t_param params, t_speed* cells, t_speed* tmp_cells) {
  /* loop over _all_ cells */
  #pragma omp parallel for
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      cells->speeds[0][ii  + jj *params.nx] = tmp_cells->speeds[0][ii + jj*params.nx]; /* central cell, no movement */
      cells->speeds[1][x_e + jj *params.nx] = tmp_cells->speeds[1][ii + jj*params.nx]; /* east */
      cells->speeds[2][ii  + y_n*params.nx] = tmp_cells->speeds[2][ii + jj*params.nx]; /* north */
      cells->speeds[3][x_w + jj *params.nx] = tmp_cells->speeds[3][ii + jj*params.nx]; /* west */
      cells->speeds[4][ii  + y_s*params.nx] = tmp_cells->speeds[4][ii + jj*params.nx]; /* south */
      cells->speeds[5][x_e + y_n*params.nx] = tmp_cells->speeds[5][ii + jj*params.nx]; /* north-east */
      cells->speeds[6][x_w + y_n*params.nx] = tmp_cells->speeds[6][ii + jj*params.nx]; /* north-west */
      cells->speeds[7][x_w + y_s*params.nx] = tmp_cells->speeds[7][ii + jj*params.nx]; /* south-west */
      cells->speeds[8][x_e + y_s*params.nx] = tmp_cells->speeds[8][ii + jj*params.nx]; /* south-east */
    }
  }

  return EXIT_SUCCESS;
}

/*
** Work with boundary conditions. The upper and lower boundaries use the rebound plane, 
** the left border is the inlet of fixed speed, and 
** the right border is the open outlet of the first-order approximation.
*/
int boundary(const t_param params, t_speed* cells,  t_speed* tmp_cells, float* inlets) {
  /* Set the constant coefficient */
  const float cst1 = 2.0/3.0;
  const float cst2 = 1.0/6.0;
  const float cst3 = 1.0/2.0;

  int ii, jj; 
  float local_density;
  
  // top wall (bounce)
  jj = params.ny -1;
  for(ii = 0; ii < params.nx; ii++){
    cells->speeds[4][ii + jj*params.nx] = tmp_cells->speeds[2][ii + jj*params.nx];
    cells->speeds[7][ii + jj*params.nx] = tmp_cells->speeds[5][ii + jj*params.nx];
    cells->speeds[8][ii + jj*params.nx] = tmp_cells->speeds[6][ii + jj*params.nx];
  }

  // bottom wall (bounce)
  jj = 0;
  for(ii = 0; ii < params.nx; ii++){
    cells->speeds[2][ii + jj*params.nx] = tmp_cells->speeds[4][ii + jj*params.nx];
    cells->speeds[5][ii + jj*params.nx] = tmp_cells->speeds[7][ii + jj*params.nx];
    cells->speeds[6][ii + jj*params.nx] = tmp_cells->speeds[8][ii + jj*params.nx];
  }

  // left wall (inlet)
  ii = 0;
  for(jj = 0; jj < params.ny; jj++){
    local_density = ( cells->speeds[0][ii + jj*params.nx]
                      + cells->speeds[2][ii + jj*params.nx]
                      + cells->speeds[4][ii + jj*params.nx]
                      + 2.0 * cells->speeds[3][ii + jj*params.nx]
                      + 2.0 * cells->speeds[6][ii + jj*params.nx]
                      + 2.0 * cells->speeds[7][ii + jj*params.nx]
                      )/(1.0 - inlets[jj]);

    cells->speeds[1][ii + jj*params.nx] = cells->speeds[3][ii + jj*params.nx]
                                        + cst1*local_density*inlets[jj];

    cells->speeds[5][ii + jj*params.nx] = cells->speeds[7][ii + jj*params.nx]
                                        - cst3*(cells->speeds[2][ii + jj*params.nx]-cells->speeds[4][ii + jj*params.nx])
                                        + cst2*local_density*inlets[jj];

    cells->speeds[8][ii + jj*params.nx] = cells->speeds[6][ii + jj*params.nx]
                                        + cst3*(cells->speeds[2][ii + jj*params.nx]-cells->speeds[4][ii + jj*params.nx])
                                        + cst2*local_density*inlets[jj];
  
  }

  // right wall (outlet)
  ii = params.nx-1;
  for(jj = 0; jj < params.ny; jj++){

    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      cells->speeds[kk][ii + jj*params.nx] = cells->speeds[kk][ii-1 + jj*params.nx];
    }
    
  }
  
  return EXIT_SUCCESS;
}
