#include "types.h"
#include "string.h"
#include "mm_malloc.h"

#define ALIGNED 32

/* utility functions */
void die(const char *message, const int line, const char *file) {
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char *paramfile, const char *obstaclefile,
               t_param *params_ptr, t_speed *cells,
               float **obstacles_ptr, float **inlets_ptr) {
  char   message[1024];  /* message buffer */
  FILE *fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter nx */
  retval = fscanf(fp, "nx: %d\n", &(params_ptr->nx));
  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  /* read in the parameter ny */
  retval = fscanf(fp, "ny: %d\n", &(params_ptr->ny));
  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  /* read in the parameter maxIters */
  retval = fscanf(fp, "iters: %d\n", &(params_ptr->maxIters));
  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  /* read in the parameter density */
  retval = fscanf(fp, "density: %f\n", &(params_ptr->density));
  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  /* read in the parameter viscosity */
  retval = fscanf(fp, "viscosity: %f\n", &(params_ptr->viscosity));
  if (retval != 1) die("could not read param file: viscosity", __LINE__, __FILE__);

  /* read in the parameter velocity */
  retval = fscanf(fp, "velocity: %f\n", &(params_ptr->velocity));
  if (retval != 1) die("could not read param file: velocity", __LINE__, __FILE__);

  /* read in the parameter type */
  retval = fscanf(fp, "type: %d\n", &(params_ptr->type));
  if (retval != 1) die("could not read param file: type", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /* calculation of relaxtion parameter */
  params_ptr->omega = 1. / (3. * params_ptr->viscosity + 0.5);

  /* Check the calculation stability */
  if (params_ptr->velocity > 0.2)
    printf("Warning: There maybe computational instability due to compressibility.\n");
  if ((2 - params_ptr->omega) < 0.15)
    printf("Warning: Possible divergence of results due to relaxation time.\n");


  t_param params = *params_ptr;
  /* Allocate memory. */

  /* main grid */
  // maxIters
  cells->speeds[0] = _mm_malloc(sizeof(float) * (params.ny * params.nx), ALIGNED);
  if (cells->speeds[0] == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  cells->speeds[1] = _mm_malloc(sizeof(float) * (params.ny * params.nx + params.maxIters), ALIGNED);
  if (cells->speeds[1] == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  cells->speeds[2] = _mm_malloc(sizeof(float) * ((params.ny + params.maxIters) * params.nx), ALIGNED);
  if (cells->speeds[2] == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  cells->speeds[3] = _mm_malloc(sizeof(float) * (params.ny * params.nx + params.maxIters), ALIGNED);
  if (cells->speeds[3] == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  cells->speeds[4] = _mm_malloc(sizeof(float) * ((params.ny + params.maxIters) * params.nx), ALIGNED);
  if (cells->speeds[4] == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  cells->speeds[5] = _mm_malloc(sizeof(float) * (params.ny * params.nx + params.maxIters * (params.nx + 1)), ALIGNED);
  if (cells->speeds[5] == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  cells->speeds[6] = _mm_malloc(sizeof(float) * (params.ny * params.nx + params.maxIters * (params.nx - 1)), ALIGNED);
  if (cells->speeds[6] == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  cells->speeds[7] = _mm_malloc(sizeof(float) * (params.ny * params.nx + params.maxIters * (params.nx + 1)), ALIGNED);
  if (cells->speeds[7] == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  cells->speeds[8] = _mm_malloc(sizeof(float) * (params.ny * params.nx + params.maxIters * (params.nx - 1)), ALIGNED);
  if (cells->speeds[8] == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  for (int k = 0; k < NSPEEDS; k++)
    printf("speeds[%d]: %p\n", k, cells->speeds[k]);

  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(float) * (params.ny * params.nx), ALIGNED);
  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params_ptr->density * 4.f / 9.f;
  float w1 = params_ptr->density / 9.f;
  float w2 = params_ptr->density / 36.f;

  for (int i = 0; i < params.nx * params.ny; i++)
    cells->speeds[0][i] = w0;

  for (int i = 0; i < params.nx * params.ny + params.maxIters; i++)
    cells->speeds[1][i] = w1;

  for (int i = 0; i < params.nx * (params.ny + params.maxIters); i++)
    cells->speeds[2][i] = w1;

  for (int i = 0; i < params.nx * params.ny + params.maxIters; i++)
    cells->speeds[3][i] = w1;

  for (int i = 0; i < params.nx * (params.ny + params.maxIters); i++)
    cells->speeds[4][i] = w1;

  for (int i = 0; i < params.ny * params.nx + params.maxIters * (params.nx + 1); i++)
    cells->speeds[5][i] = w2;

  for (int i = 0; i < params.ny * params.nx + params.maxIters * (params.nx - 1); i++)
    cells->speeds[6][i] = w2;

  for (int i = 0; i < params.ny * params.nx + params.maxIters * (params.nx + 1); i++)
    cells->speeds[7][i] = w2;

  for (int i = 0; i < params.ny * params.nx + params.maxIters * (params.nx - 1); i++)
    cells->speeds[8][i] = w2;

  cells->speeds[1] += params.maxIters;
  cells->speeds[2] += params.maxIters * params.nx;
  cells->speeds[5] += params.maxIters * (params.nx + 1);
  cells->speeds[6] += params.maxIters * (params.nx - 1);

  /* first set all cells in obstacle array to zero */
  for (int i = 0; i < params.ny * params.nx; i++)
    (*obstacles_ptr)[i] = 0;

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
    /* Center the obstacle on the y-axis. */
    yy = yy + params.ny / 2;

    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx<0 || xx > params.nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy<0 || yy > params.ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy * params.nx] = -blocked;
  }

  /* and close the file */
  fclose(fp);

  /* allocate space to hold the velocity of the cells at the inlet. */
  *inlets_ptr = (float *)_mm_malloc(sizeof(float) * params.ny, ALIGNED);

  return EXIT_SUCCESS;
}

/* finalise, including _mm_freeing up allocated memory */
int finalise(const t_param *params, t_speed *cells,
             float **obstacles_ptr, float **inlets) {
  /*
  ** _mm_free up allocated memory
  */
  cells->speeds[3] -= params->maxIters;
  cells->speeds[4] -= params->nx * params->maxIters;
  cells->speeds[7] -= (params->nx + 1) * params->maxIters;
  cells->speeds[8] -= (params->nx - 1) * params->maxIters;
  for (int k = 0; k < NSPEEDS; k++) {
    printf("speeds[%d]: %p\n", k, cells->speeds[k]);
    _mm_free(cells->speeds[k]);
  }

  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  _mm_free(*inlets);
  *inlets = NULL;

  return EXIT_SUCCESS;
}


/* write state of current grid */
int write_state(char *filename, const t_param params, t_speed *cells, float *obstacles) {
  FILE *fp;                    /* file pointer */
  float local_density;         /* per grid cell sum of densities */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(filename, "w");

  if (fp == NULL) {
    printf("%s\n", filename);
    die("could not open file output file", __LINE__, __FILE__);
  }

  /* loop on grid to calculate the velocity of each cell */
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      if (obstacles[ii + jj * params.nx]) { /* an obstacle cell */
        u = -0.05f;
      }
      else { /* no obstacle */
        local_density = 0.f;

        local_density += cells->speeds[0][ii + jj * params.nx];
        local_density += cells->speeds[1][ii + jj * params.nx];
        local_density += cells->speeds[2][ii + jj * params.nx];
        local_density += cells->speeds[3][ii + jj * params.nx];
        local_density += cells->speeds[4][ii + jj * params.nx];
        local_density += cells->speeds[5][ii + jj * params.nx];
        local_density += cells->speeds[6][ii + jj * params.nx];
        local_density += cells->speeds[7][ii + jj * params.nx];
        local_density += cells->speeds[8][ii + jj * params.nx];

        /* compute x velocity component */
        u_x = (cells->speeds[1][ii + jj * params.nx]
               + cells->speeds[5][ii + jj * params.nx]
               + cells->speeds[8][ii + jj * params.nx]
               - (cells->speeds[3][ii + jj * params.nx]
                  + cells->speeds[6][ii + jj * params.nx]
                  + cells->speeds[7][ii + jj * params.nx]))
          / local_density;
        /* compute y velocity component */
        u_y = (cells->speeds[2][ii + jj * params.nx]
               + cells->speeds[5][ii + jj * params.nx]
               + cells->speeds[6][ii + jj * params.nx]
               - (cells->speeds[4][ii + jj * params.nx]
                  + cells->speeds[7][ii + jj * params.nx]
                  + cells->speeds[8][ii + jj * params.nx]))
          / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E\n", ii, jj, u);
    }
  }

  /* close file */
  fclose(fp);

  return EXIT_SUCCESS;
}