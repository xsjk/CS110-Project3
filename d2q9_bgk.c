#include "d2q9_bgk.h"
#include <omp.h>
#include <immintrin.h>

#define TILE_SIZE 16

/* Create a selector for use with the SHUFPS instruction.  */
#define _MM_SHUFFLE(fp3,fp2,fp1,fp0) \
 (((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | (fp0))

/* The main processes in one step */
int streaming_and_boundary(const t_param params, t_speed* restrict cells, const t_speed* restrict tmp_cells, const float* restrict inlets, const int n_iter);
int collision_and_obstacle(const t_param params, const t_speed* restrict cells, t_speed* restrict tmp_cells, const int* restrict obstacles, const int n_iter);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** collision(), obstacle(), streaming() & boundary()
*/
int timestep(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, const float* restrict inlets, const int* restrict obstacles, const int n_iter)
{
  collision_and_obstacle(params, cells, tmp_cells, obstacles, n_iter);
  streaming_and_boundary(params, cells, tmp_cells, inlets, n_iter);
  return EXIT_SUCCESS;
}


int collision_and_obstacle(const t_param params, const t_speed* restrict cells, t_speed* restrict tmp_cells, const int* restrict obstacles, const int n_iter) {
  /* All the const quantities are unnecessary 
   * since the compiler is smart enough. */
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */
  const float inv_c_sq = 1.f / c_sq;
  const float inv_2c_sq2 = 1.f / (2.f * c_sq * c_sq);
  
  /* Althought _mm256_set1_ps (sequential) is slower than _mm256_broadcast_ss (parallel) 
   * Since the value can be determined at compile time, it is not necessary to use broadcast. */
  const __m256 inv_c_sqv = _mm256_broadcast_ss(&inv_c_sq);
  const __m256 inv_2c_sq2v = _mm256_broadcast_ss(&inv_2c_sq2);
  const __m256 onev = _mm256_set1_ps(1.f);
  const __m128 w1v = _mm_broadcast_ss(&w1);
  const __m128 w2v = _mm_broadcast_ss(&w2);
  const __m256 w12v = _mm256_insertf128_ps(_mm256_castps128_ps256(w1v), w2v, 1);

  /* loop over the cells in the grid
  ** the collision step is called before
  ** the streaming step and so values of interest
  ** are in the scratch-space grid */
  #pragma omp parallel for schedule(dynamic)
  // for (int jj = 0; jj < params.ny; jj += 1)
  //   for (int ii = 0; ii < params.nx; ii += 1) {
  for (int j = 0; j < params.ny; j += TILE_SIZE)
    for (int i = 0; i < params.nx && i <= n_iter; i += TILE_SIZE)
      for (int jj = j; jj < j + TILE_SIZE && jj < params.ny; jj++)
        for (int ii = i; ii < i + TILE_SIZE && ii < params.nx; ii++) {
          if (__builtin_expect(obstacles[jj*params.nx + ii], 0)) 

          /*
          ** For obstacles, mirror their speed.
          */
          {
            /* called after collision, so taking values from scratch space
            ** mirroring, and writing into main grid */
            tmp_cells[ii + jj*params.nx].speeds[0] = cells[ii + jj*params.nx].speeds[0];
            const __m256 cellsv = _mm256_loadu_ps(cells[ii + jj*params.nx].speeds + 1);
            const __m256 tmp_cellsv = _mm256_permute_ps(cellsv, _MM_SHUFFLE(1, 0, 3, 2));
            _mm256_storeu_ps(tmp_cells[ii + jj*params.nx].speeds + 1, tmp_cellsv);
          } 

          else 

          /*
          ** The collision of fluids in the cell is calculated using 
          ** the local equilibrium distribution and relaxation process
          */

          {
            /* compute local density total */
            #ifndef SIMD_SUM8
              float local_density = 0.f;
              for (int kk = 0; kk < NSPEEDS; kk++)
                local_density += cells[ii + jj*params.nx].speeds[kk];
            #else
              __m256 speeds;
              speeds = _mm256_loadu_ps(cells[ii + jj*params.nx].speeds + 1);
              speeds = _mm256_add_ps(speeds, _mm256_permute2f128_ps(speeds, speeds, 1));
              speeds = _mm256_add_ps(speeds, _mm256_shuffle_ps(speeds, speeds, _MM_SHUFFLE(1, 0, 3, 2)));
              speeds = _mm256_add_ps(speeds, _mm256_shuffle_ps(speeds, speeds, _MM_SHUFFLE(2, 3, 0, 1)));
              const float local_density = _mm256_cvtss_f32(speeds) + cells[ii + jj*params.nx].speeds[0];
            #endif
            const __m256 local_densityv = _mm256_broadcast_ss(&local_density);

            /* compute x velocity component */
            const float u_x = (cells[ii + jj*params.nx].speeds[1]
                          + cells[ii + jj*params.nx].speeds[5]
                          + cells[ii + jj*params.nx].speeds[8]
                          - (cells[ii + jj*params.nx].speeds[3]
                            + cells[ii + jj*params.nx].speeds[6]
                            + cells[ii + jj*params.nx].speeds[7]))
                        / local_density;
            const __m256 u_xv = _mm256_broadcast_ss(&u_x);
            
            /* compute y velocity component */
            const float u_y = (cells[ii + jj*params.nx].speeds[2]
                          + cells[ii + jj*params.nx].speeds[5]
                          + cells[ii + jj*params.nx].speeds[6]
                          - (cells[ii + jj*params.nx].speeds[4]
                            + cells[ii + jj*params.nx].speeds[7]
                            + cells[ii + jj*params.nx].speeds[8]))
                        / local_density;
            const __m256 u_yv = _mm256_broadcast_ss(&u_y);

            /* velocity squared */
            const float u_sq = u_x * u_x + u_y * u_y;
            const float uu_sq = u_sq / (2.f * c_sq);
            const __m256 uu_sqv = _mm256_broadcast_ss(&uu_sq);

            /* directional velocity components */
            const __m256 _ku_x = _mm256_set_ps( 1, -1,-1, 1, 0,-1, 0, 1);
            const __m256 _ku_y = _mm256_set_ps(-1, -1, 1, 1,-1, 0, 1, 0);
            const __m256 uv = _mm256_fmadd_ps(_ku_y, u_yv, _mm256_mul_ps(_ku_x, u_xv));
            const __m256 u2v = _mm256_mul_ps(uv, uv);
            
            /* equilibrium densities */
            tmp_cells[ii + jj*params.nx].speeds[0] = cells[ii + jj*params.nx].speeds[0]
                                                    + params.omega
                                                    * ((w0 * local_density * (1.f - uu_sq)) 
                                                      - cells[ii + jj*params.nx].speeds[0]);
            const __m256 _1 = _mm256_fmadd_ps(uv, inv_c_sqv, onev);
            const __m256 _2 = _mm256_fmsub_ps(u2v, inv_2c_sq2v, uu_sqv);
            const __m256 _3 = _mm256_mul_ps(w12v, local_densityv);
            const __m256 d_equv = _mm256_mul_ps(_3, _mm256_add_ps(_1, _2));
            
            /* relaxation step */
            const __m256 omegav = _mm256_broadcast_ss(&params.omega);
            const __m256 cellsv = _mm256_loadu_ps(cells[ii + jj*params.nx].speeds+1);
            const __m256 tmp_cellsv = _mm256_fmadd_ps(omegav, _mm256_sub_ps(d_equv, cellsv), cellsv);
            _mm256_storeu_ps(tmp_cells[ii + jj*params.nx].speeds+1, tmp_cellsv);
            
          }
    }
    
  
  return EXIT_SUCCESS;
}



/*
** Work with boundary conditions. The upper and lower boundaries use the rebound plane, 
** the left border is the inlet of fixed speed, and 
** the right border is the open outlet of the first-order approximation.
*/

int streaming_and_boundary(const t_param params, t_speed* restrict cells, const t_speed* restrict tmp_cells, const float* restrict inlets, const int n_iter) {
  /* Set the constant coefficient */
  const float cst1 = 2.0/3.0;
  const float cst2 = 1.0/6.0;
  const float cst3 = 1.0/2.0;

  register int ii, jj;
  register float local_density;
  
  #pragma omp parallel for
  for (int j = 1; j < params.ny-1; j += TILE_SIZE)
    for (int i = 1; i < params.nx-1 && i <= n_iter; i += TILE_SIZE)
      for (int jj = j; jj < j + TILE_SIZE && jj < params.ny-1; jj++)
        for (int ii = i; ii < i + TILE_SIZE && ii < params.nx-1; ii++) {
          /* determine indices of axis-direction neighbours
          ** respecting periodic boundary conditions (wrap around) */
          int y_n = jj + 1;
          int x_e = ii + 1;
          int y_s = jj - 1;
          int x_w = ii - 1;
          /* propagate densities from neighbouring cells, following
          ** appropriate directions of travel and writing into
          ** scratch space grid */
          cells[ii + jj*params.nx].speeds[0] = tmp_cells[ii  + jj *params.nx].speeds[0]; /* central cell, no movement */
          cells[ii + jj*params.nx].speeds[1] = tmp_cells[x_w + jj *params.nx].speeds[1]; /* east */
          cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii  + y_s*params.nx].speeds[2]; /* north */
          cells[ii + jj*params.nx].speeds[3] = tmp_cells[x_e + jj *params.nx].speeds[3]; /* west */
          cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii  + y_n*params.nx].speeds[4]; /* south */
          cells[ii + jj*params.nx].speeds[5] = tmp_cells[x_w + y_s*params.nx].speeds[5]; /* north-east */
          cells[ii + jj*params.nx].speeds[6] = tmp_cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
          cells[ii + jj*params.nx].speeds[7] = tmp_cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
          cells[ii + jj*params.nx].speeds[8] = tmp_cells[x_w + y_n*params.nx].speeds[8]; /* south-east */                 
        }

  for(ii = 1; ii < params.nx-1 && ii <= n_iter; ii++) {
    int x_e = ii + 1;
    int x_w = ii - 1;
    /* bottom wall */
    jj = 0;
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = jj + 1;
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      
      cells[ii + jj*params.nx].speeds[0] = tmp_cells[ii  + jj *params.nx].speeds[0]; /* central cell, no movement */
      cells[ii + jj*params.nx].speeds[1] = tmp_cells[x_w + jj *params.nx].speeds[1]; /* east */
      cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii  + jj *params.nx].speeds[4]; /* north */
      cells[ii + jj*params.nx].speeds[3] = tmp_cells[x_e + jj *params.nx].speeds[3]; /* west */
      cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii  + y_n*params.nx].speeds[4]; /* south */
      cells[ii + jj*params.nx].speeds[5] = tmp_cells[ii  + jj *params.nx].speeds[7]; /* north-east */
      cells[ii + jj*params.nx].speeds[6] = tmp_cells[ii  + jj *params.nx].speeds[8]; /* north-west */
      cells[ii + jj*params.nx].speeds[7] = tmp_cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
      cells[ii + jj*params.nx].speeds[8] = tmp_cells[x_w + y_n*params.nx].speeds[8]; /* south-east */ 
    }
    /* top wall */
    jj = params.ny - 1;
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_s = jj - 1;
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */

      cells[ii + jj*params.nx].speeds[0] = tmp_cells[ii  + jj *params.nx].speeds[0]; /* central cell, no movement */
      cells[ii + jj*params.nx].speeds[1] = tmp_cells[x_w + jj *params.nx].speeds[1]; /* east */
      cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii  + y_s*params.nx].speeds[2]; /* north */
      cells[ii + jj*params.nx].speeds[3] = tmp_cells[x_e + jj *params.nx].speeds[3]; /* west */
      cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii  + jj *params.nx].speeds[2]; /* south */
      cells[ii + jj*params.nx].speeds[5] = tmp_cells[x_w + y_s*params.nx].speeds[5]; /* north-east */
      cells[ii + jj*params.nx].speeds[6] = tmp_cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
      cells[ii + jj*params.nx].speeds[7] = tmp_cells[ii  + jj *params.nx].speeds[5]; /* south-west */
      cells[ii + jj*params.nx].speeds[8] = tmp_cells[ii  + jj *params.nx].speeds[6]; /* south-east */                
      
    }
  }

  #pragma omp parallel for
  for(jj = 1; jj < params.ny-1; jj++) {
    int y_n = jj + 1;
    int y_s = jj - 1;
    /* left wall */
    ii = 0;
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int x_e = ii + 1;
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      
      cells[ii + jj*params.nx].speeds[0] = tmp_cells[ii  + jj *params.nx].speeds[0]; /* central cell, no movement */
      cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii  + y_s*params.nx].speeds[2]; /* north */
      cells[ii + jj*params.nx].speeds[3] = tmp_cells[x_e + jj *params.nx].speeds[3]; /* west */
      cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii  + y_n*params.nx].speeds[4]; /* south */
      cells[ii + jj*params.nx].speeds[6] = tmp_cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
      cells[ii + jj*params.nx].speeds[7] = tmp_cells[x_e + y_n*params.nx].speeds[7]; /* south-west */

      local_density = ( cells[ii + jj*params.nx].speeds[0]
                    + cells[ii + jj*params.nx].speeds[2]
                    + cells[ii + jj*params.nx].speeds[4]
                    + 2.0 * cells[ii + jj*params.nx].speeds[3]
                    + 2.0 * cells[ii + jj*params.nx].speeds[6]
                    + 2.0 * cells[ii + jj*params.nx].speeds[7]
                    )/(1.0 - inlets[jj]);

      cells[ii + jj*params.nx].speeds[1] = cells[ii + jj*params.nx].speeds[3]
                                          + cst1*local_density*inlets[jj];

      cells[ii + jj*params.nx].speeds[5] = cells[ii + jj*params.nx].speeds[7]
                                          - cst3*(cells[ii + jj*params.nx].speeds[2]-cells[ii + jj*params.nx].speeds[4])
                                          + cst2*local_density*inlets[jj];

      cells[ii + jj*params.nx].speeds[8] = cells[ii + jj*params.nx].speeds[6]
                                          + cst3*(cells[ii + jj*params.nx].speeds[2]-cells[ii + jj*params.nx].speeds[4])
                                          + cst2*local_density*inlets[jj];

    }
    /* right wall */
    ii = params.nx - 1;
    if (ii <= n_iter) {
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      cells[ii + jj*params.nx].speeds[0] = cells[ii-1 + jj*params.nx].speeds[0];
      _mm256_storeu_ps(cells[ii + jj*params.nx].speeds+1, _mm256_loadu_ps(cells[ii-1 + jj*params.nx].speeds+1));
    }
  }


  /* bottom left */ 
  ii = 0;
  jj = 0;
  {
    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int y_n = jj + 1;
    int x_e = ii + 1;
    /* propagate densities from neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
   
      
    cells[ii + jj*params.nx].speeds[0] = tmp_cells[ii  + jj *params.nx].speeds[0]; /* central cell, no movement */
    cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii  + jj *params.nx].speeds[4]; /* north */
    cells[ii + jj*params.nx].speeds[3] = tmp_cells[x_e + jj *params.nx].speeds[3]; /* west */
    cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii  + y_n*params.nx].speeds[4]; /* south */
    cells[ii + jj*params.nx].speeds[6] = tmp_cells[ii  + jj *params.nx].speeds[8]; /* north-west */
    cells[ii + jj*params.nx].speeds[7] = tmp_cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
    
    
    local_density = ( cells[ii + jj*params.nx].speeds[0]
                      + cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[4]
                      + 2.0 * cells[ii + jj*params.nx].speeds[3]
                      + 2.0 * cells[ii + jj*params.nx].speeds[6]
                      + 2.0 * cells[ii + jj*params.nx].speeds[7]
                      )/(1.0 - inlets[jj]);

    cells[ii + jj*params.nx].speeds[1] = cells[ii + jj*params.nx].speeds[3]
                                        + cst1*local_density*inlets[jj];

    cells[ii + jj*params.nx].speeds[5] = cells[ii + jj*params.nx].speeds[7]
                                        - cst3*(cells[ii + jj*params.nx].speeds[2]-cells[ii + jj*params.nx].speeds[4])
                                        + cst2*local_density*inlets[jj];

    cells[ii + jj*params.nx].speeds[8] = cells[ii + jj*params.nx].speeds[6]
                                        + cst3*(cells[ii + jj*params.nx].speeds[2]-cells[ii + jj*params.nx].speeds[4])
                                        + cst2*local_density*inlets[jj];
  

  }
  /* bottom right */
  ii = params.nx - 1;
  jj = 0;
  if (ii <= n_iter) {
    /* propagate densities from neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
    cells[ii + jj*params.nx].speeds[0] = cells[ii-1 + jj*params.nx].speeds[0];
    _mm256_storeu_ps(cells[ii + jj*params.nx].speeds+1, _mm256_loadu_ps(cells[ii-1 + jj*params.nx].speeds+1));

  }
  /* top left */
  ii = 0;
  jj = params.ny - 1;
  {
    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int x_e = ii + 1;
    int y_s = jj - 1;
    /* propagate densities from neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
   
    cells[ii + jj*params.nx].speeds[0] = tmp_cells[ii  + jj *params.nx].speeds[0]; /* central cell, no movement */
    cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii  + y_s*params.nx].speeds[2]; /* north */
    cells[ii + jj*params.nx].speeds[3] = tmp_cells[x_e + jj *params.nx].speeds[3]; /* west */
    cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii  + jj *params.nx].speeds[2]; /* south */
    cells[ii + jj*params.nx].speeds[6] = tmp_cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
    cells[ii + jj*params.nx].speeds[7] = tmp_cells[ii  + jj *params.nx].speeds[5]; /* south-west */

    
    local_density = ( cells[ii + jj*params.nx].speeds[0]
                      + cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[4]
                      + 2.0 * cells[ii + jj*params.nx].speeds[3]
                      + 2.0 * cells[ii + jj*params.nx].speeds[6]
                      + 2.0 * cells[ii + jj*params.nx].speeds[7]
                      )/(1.0 - inlets[jj]);

    cells[ii + jj*params.nx].speeds[1] = cells[ii + jj*params.nx].speeds[3]
                                        + cst1*local_density*inlets[jj];

    cells[ii + jj*params.nx].speeds[5] = cells[ii + jj*params.nx].speeds[7]
                                        - cst3*(cells[ii + jj*params.nx].speeds[2]-cells[ii + jj*params.nx].speeds[4])
                                        + cst2*local_density*inlets[jj];

    cells[ii + jj*params.nx].speeds[8] = cells[ii + jj*params.nx].speeds[6]
                                        + cst3*(cells[ii + jj*params.nx].speeds[2]-cells[ii + jj*params.nx].speeds[4])
                                        + cst2*local_density*inlets[jj];
  
  }
  /* top right */
  ii = params.nx - 1;
  jj = params.ny - 1;
  if (ii <= n_iter) {
    /* propagate densities from neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
    cells[ii + jj*params.nx].speeds[0] = cells[ii-1 + jj*params.nx].speeds[0];
    _mm256_storeu_ps(cells[ii + jj*params.nx].speeds+1, _mm256_loadu_ps(cells[ii-1 + jj*params.nx].speeds+1));
  }
  
  return EXIT_SUCCESS;
}
