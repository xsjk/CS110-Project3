#include "calc.h"

/* set inlets velocity there are two type inlets*/
int set_inlets(const t_param params, float* inlets) {
  for(int jj=0; jj <params.ny; jj++){
    if(!params.type)
      inlets[jj]=params.velocity; // homogeneous
    else
      inlets[jj]=params.velocity * 4.0 *((1-((float)jj)/params.ny)*((float)(jj+1))/params.ny); // parabolic
  }
  return EXIT_SUCCESS;
}

/* compute average velocity of whole grid, ignore grids with obstacles. */
float av_velocity(const t_param params, t_speed* cells, float* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float  tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        local_density += cells->speeds[0][ii + jj*params.nx];
        local_density += cells->speeds[1][ii + jj*params.nx];
        local_density += cells->speeds[2][ii + (params.ny - 1 - jj) * params.nx];
        local_density += cells->speeds[3][ii + jj*params.nx];
        local_density += cells->speeds[4][ii + jj*params.nx];
        local_density += cells->speeds[5][(params.ny + ii - jj) + (params.ny - 1 - jj)*(params.nx + params.ny)];
        local_density += cells->speeds[6][(ii + jj) + (params.ny - 1 - jj)*(params.nx + params.ny)];
        local_density += cells->speeds[7][ii + jj*params.nx];
        local_density += cells->speeds[8][ii + jj*params.nx];
        

        /* x-component of velocity */
        float u_x = (cells->speeds[1][ii + jj*params.nx]
                      + cells->speeds[5][(params.ny + ii - jj) + (params.ny - 1 - jj)*(params.nx + params.ny)]
                      + cells->speeds[8][ii + jj*params.nx]
                      - (cells->speeds[3][ii + jj*params.nx]
                         + cells->speeds[6][(ii + jj) + (params.ny - 1 - jj)*(params.nx + params.ny)]
                         + cells->speeds[7][ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells->speeds[2][ii + (params.ny - 1 - jj) * params.nx]
                      + cells->speeds[5][(params.ny + ii - jj) + (params.ny - 1 - jj)*(params.nx + params.ny)]
                      + cells->speeds[6][(ii + jj) + (params.ny - 1 - jj)*(params.nx + params.ny)]
                      - (cells->speeds[4][ii + jj*params.nx]
                         + cells->speeds[7][ii + jj*params.nx]
                         + cells->speeds[8][ii + jj*params.nx]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

/* calculate reynold number */
float calc_reynolds(const t_param params, t_speed* cells, float* obstacles)
{
  return av_velocity(params, cells, obstacles) * (float)(params.ny) / params.viscosity;
}