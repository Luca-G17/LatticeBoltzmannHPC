/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle fileaccelerate_flow.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <omp.h>
#include <malloc.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
  int neighbours[4]; // n, e, s, w
} t_speed;

typedef struct 
{
  float *restrict speeds[NSPEEDS]; /* Array of arrays of speeds for each cell - speeds[k][i] = kth speed for cell i  */
  int *restrict neighbours[4]; /* Array of arrays of neighbors for each cell */
  char *restrict obstacle;
} SoAGrid;

typedef struct
{
  float *restrict speeds;
  int *restrict neighbours;
} ArrayGrid;

typedef struct 
{
  int *restrict cell_ptrs;
  unsigned int size;
} obstacle_loc_array;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, ArrayGrid*restrict* cells_ptr, ArrayGrid*restrict* tmp_cells_ptr,
               int*restrict* obstacles_ptr, float** av_vels_ptr, SoAGrid*restrict* grid_ptr, SoAGrid*restrict* tmp_grid_ptr, t_speed*restrict* aos_grid_ptr, t_speed*restrict* tmp_aos_grid_ptr, obstacle_loc_array*restrict* obst_locs, float* tot_cells);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, const ArrayGrid *restrict cells, const ArrayGrid *restrict tmp_cells, int *restrict obstacles, const obstacle_loc_array*restrict obst_locs);
float timestep_SoA(const t_param params, const SoAGrid *restrict grid, const SoAGrid *restrict tmp_grid, const obstacle_loc_array *restrict obst_locs, const int *restrict obstacles);
float timestep_AoS(const t_param params, t_speed *restrict aos_grid, t_speed *restrict tmp_aos_grid, const int *restrict obstacles);
int write_values(const t_param params, ArrayGrid* cells, int* obstacles, float* av_vels, int soa, SoAGrid* grid, t_speed* aos_grid);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, ArrayGrid*restrict* cells_ptr, ArrayGrid*restrict* tmp_cells_ptr,
             int*restrict* obstacles_ptr, float** av_vels_ptr, SoAGrid*restrict* grid_ptr, SoAGrid*restrict* tmp_grid_ptr, t_speed*restrict* aos_grid_ptr, t_speed*restrict* tmp_aos_grid_ptr, obstacle_loc_array*restrict* obst_locs);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float final_av_vel);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  ArrayGrid *restrict cells     = NULL;    /* grid containing fluid densities */
  ArrayGrid *restrict tmp_cells = NULL;    /* scratch space */
  t_speed   *restrict aos_grid = NULL;
  t_speed   *restrict tmp_aos_grid = NULL;
  int       *restrict obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */
  SoAGrid* grid     = NULL;
  SoAGrid* tmp_grid = NULL;
  obstacle_loc_array *restrict obst_locs = NULL;
  float tot_cells = 0;
  int SoA = 0; // 0 = False

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &grid, &tmp_grid, &aos_grid, &tmp_aos_grid, &obst_locs, &tot_cells);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  if (SoA == 0) {
    for (int tt = 0; tt < params.maxIters; tt++)
    {
      av_vels[tt] = timestep(params, cells, tmp_cells, obstacles, obst_locs) / tot_cells;
      ArrayGrid* tmp = tmp_cells;
      tmp_cells = cells;
      cells = tmp; 

#ifdef DEBUG
      printf("==timestep: %d==\n", tt);
      printf("av velocity: %.12E\n", av_vels[tt]);
      printf("tot density: %.12E\n", total_density(params, cells));
#endif
    }
  }
  else if (SoA == 1) {
    for (int tt = 0; tt < params.maxIters; tt++)
    {
      av_vels[tt] = timestep_SoA(params, grid, tmp_grid, obst_locs, obstacles) / tot_cells;
      SoAGrid* tmp = tmp_grid;
      tmp_grid = grid;
      grid = tmp;

#ifdef DEBUG
      printf("==timestep: %d==\n", tt);
      printf("av velocity: %.12E\n", av_vels[tt]);
      printf("tot density: %.12E\n", total_density(params, cells));
#endif
    }
  }
  else {
    for (int tt = 0; tt < params.maxIters; tt++) {
      av_vels[tt] = timestep_AoS(params, aos_grid, tmp_aos_grid, obstacles);
      t_speed* tmp = tmp_aos_grid;
      tmp_aos_grid = aos_grid;
      aos_grid = tmp;
    }
  }
   
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, av_vels[params.maxIters - 1]));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, cells, obstacles, av_vels, SoA, grid, aos_grid);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, &grid, &tmp_grid, &aos_grid, &tmp_aos_grid, &obst_locs);

  return EXIT_SUCCESS;
}

float timestep_AoS(const t_param params, t_speed *restrict grid, t_speed *restrict tmp_grid, const int *restrict obstacle)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float two_c_four = c_sq * c_sq * 2.0f;
  const float two_c_sq = c_sq * 2.0f;
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */
  tot_u = 0.f;

  __assume((params.nx) % 2 == 0);
  __assume((params.ny) % 2 == 0);

  const float accel_w1 = params.density * params.accel / 9.f;
  const float accel_w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  const int jj = params.ny - 2;

  #pragma omp simd
  for (int ii = 0; ii < params.nx; ii++) {
    const int cell_i = (ii + jj * params.nx);
    if (!obstacle[cell_i] && 
       (grid[cell_i].speeds[3] - accel_w1) > 0.0f &&
       (grid[cell_i].speeds[6] - accel_w2) > 0.0f &&
       (grid[cell_i].speeds[7] - accel_w2) > 0.0f) {
      
      grid[cell_i].speeds[1] += accel_w1;
      grid[cell_i].speeds[5] += accel_w2;
      grid[cell_i].speeds[8] += accel_w2;

      grid[cell_i].speeds[3] -= accel_w1;
      grid[cell_i].speeds[6] -= accel_w2;
      grid[cell_i].speeds[7] -= accel_w2;
    }
  }

  #pragma omp parallel for reduction(+:tot_cells) reduction(+:tot_u) schedule(static)
  for (int jj = 0; jj < params.ny; jj++)
  {
    // #pragma omp simd reduction(+:tot_cells) reduction(+:tot_u)
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int cell_i = (ii + jj * params.nx);
      int y_n = grid[cell_i].neighbours[0];
      int x_e = grid[cell_i].neighbours[1];
      int y_s = grid[cell_i].neighbours[2];
      int x_w = grid[cell_i].neighbours[3];
      
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      if (!obstacle[ii + jj*params.nx]) {
        tmp_grid[cell_i].speeds[0] = grid[ii + jj*params.nx].speeds[0]; /* central cell, no movement */ // TODO: Useless copy
        tmp_grid[cell_i].speeds[1] = grid[x_w + jj*params.nx].speeds[1]; /* east */
        tmp_grid[cell_i].speeds[2] = grid[ii + y_s*params.nx].speeds[2]; /* north */
        tmp_grid[cell_i].speeds[3] = grid[x_e + jj*params.nx].speeds[3]; /* west */
        tmp_grid[cell_i].speeds[4] = grid[ii + y_n*params.nx].speeds[4]; /* south */
        tmp_grid[cell_i].speeds[5] = grid[x_w + y_s*params.nx].speeds[5]; /* north-east */
        tmp_grid[cell_i].speeds[6] = grid[x_e + y_s*params.nx].speeds[6]; /* north-west */
        tmp_grid[cell_i].speeds[7] = grid[x_e + y_n*params.nx].speeds[7]; /* south-west */
        tmp_grid[cell_i].speeds[8] = grid[x_w + y_n*params.nx].speeds[8]; /* south-east */

        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_grid[cell_i].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (tmp_grid[cell_i].speeds[1]
                   + tmp_grid[cell_i].speeds[5]
                   + tmp_grid[cell_i].speeds[8]
                  - (tmp_grid[cell_i].speeds[3]
                   + tmp_grid[cell_i].speeds[6]
                   + tmp_grid[cell_i].speeds[7]))
                   / local_density;
        /* compute y velocity component */
        float u_y = (tmp_grid[cell_i].speeds[2]
                   + tmp_grid[cell_i].speeds[5]
                   + tmp_grid[cell_i].speeds[6]
                  - (tmp_grid[cell_i].speeds[4]
                   + tmp_grid[cell_i].speeds[7]
                   + tmp_grid[cell_i].speeds[8]))
                   / local_density;

        /* velocity squared */
        float u_sq_over_two_c_sq = (u_x * u_x + u_y * u_y) / two_c_sq;

        /* directional velocity components */
        float u[NSPEEDS];
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

        d_equ[0] = w0 * local_density * (1.f - u_sq_over_two_c_sq);

        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (two_c_four)
                                         - u_sq_over_two_c_sq);
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (two_c_four)
                                         - u_sq_over_two_c_sq);
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (two_c_four)
                                         - u_sq_over_two_c_sq);
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (two_c_four)
                                         - u_sq_over_two_c_sq);
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (two_c_four)
                                         - u_sq_over_two_c_sq);
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (two_c_four)
                                         - u_sq_over_two_c_sq);
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (two_c_four)
                                         - u_sq_over_two_c_sq);
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (two_c_four)
                                         - u_sq_over_two_c_sq);

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          tmp_grid[cell_i].speeds[kk] += params.omega * (d_equ[kk] - tmp_grid[cell_i].speeds[kk]);
        }

        tot_u = tot_u + sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        tot_cells = tot_cells + 1;
      }
      else {
        // tmp_cells[ii + jj*params.nx].speeds[0] = cells[ii + jj*params.nx].speeds[0]; /* central cell, no movement */ // TODO: Useless copy
        tmp_grid[cell_i].speeds[1] = grid[x_e + jj*params.nx].speeds[3]; /* west */
        tmp_grid[cell_i].speeds[2] = grid[ii + y_n*params.nx].speeds[4]; /* south */
        tmp_grid[cell_i].speeds[3] = grid[x_w + jj*params.nx].speeds[1]; /* east */
        tmp_grid[cell_i].speeds[4] = grid[ii + y_s*params.nx].speeds[2]; /* north */
        tmp_grid[cell_i].speeds[5] = grid[x_e + y_n*params.nx].speeds[7]; /* south-west */
        tmp_grid[cell_i].speeds[6] = grid[x_w + y_n*params.nx].speeds[8]; /* south-east */
        tmp_grid[cell_i].speeds[7] = grid[x_w + y_s*params.nx].speeds[5]; /* north-east */
        tmp_grid[cell_i].speeds[8] = grid[x_e + y_s*params.nx].speeds[6]; /* north-west */
      }
    }
  }
  return tot_u / (float)tot_cells;
}

float timestep_SoA(const t_param params, const SoAGrid *restrict grid, const SoAGrid *restrict tmp_grid, const obstacle_loc_array *restrict obst_locs, const int *restrict obstacles)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float two_c_four = c_sq * c_sq * 2.0f;
  const float two_c_sq = c_sq * 2.0f;
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */
  tot_u = 0.f;

  /* loop over _all_ cells */
  for (int ss = 0; ss < 9; ss++) {
    __assume_aligned(grid->speeds[ss], 64);
    __assume_aligned(tmp_grid->speeds[ss], 64);
  }
  for (int nn = 0; nn < 4; nn++) {
    __assume_aligned(grid->neighbours[nn], 64);
    __assume_aligned(tmp_grid->neighbours[nn], 64);
  }
  __assume_aligned(grid->obstacle, 64);
  __assume_aligned(tmp_grid->obstacle, 64);

  __assume((params.nx) % 2 == 0);
  __assume((params.ny) % 2 == 0);

  const float accel_w1 = params.density * params.accel / 9.f;
  const float accel_w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  const int jj = params.ny - 2;

  #pragma omp simd
  for (int ii = 0; ii < params.nx; ii++) {
    const int cell_i = (ii + jj * params.nx);
    if (!grid->obstacle[cell_i] && 
       (grid->speeds[3][cell_i] - accel_w1) > 0.0f &&
       (grid->speeds[6][cell_i] - accel_w2) > 0.0f &&
       (grid->speeds[7][cell_i] - accel_w2) > 0.0f) {
      
      grid->speeds[1][cell_i] += accel_w1;
      grid->speeds[5][cell_i] += accel_w2;
      grid->speeds[8][cell_i] += accel_w2;

      grid->speeds[3][cell_i] -= accel_w1;
      grid->speeds[6][cell_i] -= accel_w2;
      grid->speeds[7][cell_i] -= accel_w2;
    }
  }

  // #pragma omp parallel for reduction(+:tot_cells) reduction(+:tot_u) schedule(static)
  for (int jj = 0; jj < params.ny; jj++)
  {
    // #pragma omp simd reduction(+:tot_cells) reduction(+:tot_u)
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int cell_i = (ii + jj * params.nx);
      int y_n = grid->neighbours[0][cell_i];
      int x_e = grid->neighbours[1][cell_i];
      int y_s = grid->neighbours[2][cell_i];
      int x_w = grid->neighbours[3][cell_i];
      
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_grid->speeds[0][cell_i] = grid->speeds[0][ii + jj*params.nx]; /* central cell, no movement */ // TODO: Useless copy
      tmp_grid->speeds[1][cell_i] = grid->speeds[1][x_w + jj*params.nx]; /* east */
      tmp_grid->speeds[2][cell_i] = grid->speeds[2][ii + y_s*params.nx]; /* north */
      tmp_grid->speeds[3][cell_i] = grid->speeds[3][x_e + jj*params.nx]; /* west */
      tmp_grid->speeds[4][cell_i] = grid->speeds[4][ii + y_n*params.nx]; /* south */
      tmp_grid->speeds[5][cell_i] = grid->speeds[5][x_w + y_s*params.nx]; /* north-east */
      tmp_grid->speeds[6][cell_i] = grid->speeds[6][x_e + y_s*params.nx]; /* north-west */
      tmp_grid->speeds[7][cell_i] = grid->speeds[7][x_e + y_n*params.nx]; /* south-west */
      tmp_grid->speeds[8][cell_i] = grid->speeds[8][x_w + y_n*params.nx]; /* south-east */

      /* compute local density total */
      float local_density = 0.f;
      #pragma omp simd
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        local_density += tmp_grid->speeds[kk][cell_i];
      }

      /* compute x velocity component */
      float u_x = (tmp_grid->speeds[1][cell_i]
                  + tmp_grid->speeds[5][cell_i]
                  + tmp_grid->speeds[8][cell_i]
                - (tmp_grid->speeds[3][cell_i]
                  + tmp_grid->speeds[6][cell_i]
                  + tmp_grid->speeds[7][cell_i]))
                  / local_density;
      /* compute y velocity component */
      float u_y = (tmp_grid->speeds[2][cell_i]
                  + tmp_grid->speeds[5][cell_i]
                  + tmp_grid->speeds[6][cell_i]
                - (tmp_grid->speeds[4][cell_i]
                  + tmp_grid->speeds[7][cell_i]
                  + tmp_grid->speeds[8][cell_i]))
                  / local_density;

      /* velocity squared */
      float u_sq_over_two_c_sq = (u_x * u_x + u_y * u_y) / two_c_sq;

      /* directional velocity components */
      float u[NSPEEDS];
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

      d_equ[0] = w0 * local_density * (1.f - u_sq_over_two_c_sq);

      /* axis speeds: weight w1 */
      d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                        + (u[1] * u[1]) / (two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                        + (u[2] * u[2]) / (two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                        + (u[3] * u[3]) / (two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                        + (u[4] * u[4]) / (two_c_four)
                                        - u_sq_over_two_c_sq);
      /* diagonal speeds: weight w2 */
      d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                        + (u[5] * u[5]) / (two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                        + (u[6] * u[6]) / (two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                        + (u[7] * u[7]) / (two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                        + (u[8] * u[8]) / (two_c_four)
                                        - u_sq_over_two_c_sq);

      /* relaxation step */
      #pragma omp simd
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        tmp_grid->speeds[kk][cell_i] += params.omega * (d_equ[kk] - tmp_grid->speeds[kk][cell_i]);
      }

      tot_u = tot_u + (!obstacles[cell_i] * sqrtf((u_x * u_x) + (u_y * u_y)));
    }
  }

  #pragma omp simd
  for (int ll = 0; ll < obst_locs->size; ll++) {
    const int obst_ptr = obst_locs->cell_ptrs[ll];
    const int ii = obst_ptr % params.nx;
    const int jj = obst_ptr / params.nx;
    const int cell_i = ii + (jj * params.nx);

    const int y_n = grid->neighbours[0][cell_i];
    const int x_e = grid->neighbours[1][cell_i];
    const int y_s = grid->neighbours[2][cell_i];
    const int x_w = grid->neighbours[3][cell_i];

    tmp_grid->speeds[0][cell_i] = grid->speeds[0][cell_i];
    tmp_grid->speeds[1][cell_i] = grid->speeds[3][x_e + jj*params.nx]; /* west */
    tmp_grid->speeds[2][cell_i] = grid->speeds[4][ii + y_n*params.nx]; /* south */
    tmp_grid->speeds[3][cell_i] = grid->speeds[1][x_w + jj*params.nx]; /* east */
    tmp_grid->speeds[4][cell_i] = grid->speeds[2][ii + y_s*params.nx]; /* north */
    tmp_grid->speeds[5][cell_i] = grid->speeds[7][x_e + y_n*params.nx]; /* south-west */
    tmp_grid->speeds[6][cell_i] = grid->speeds[8][x_w + y_n*params.nx]; /* south-east */
    tmp_grid->speeds[7][cell_i] = grid->speeds[5][x_w + y_s*params.nx]; /* north-east */
    tmp_grid->speeds[8][cell_i] = grid->speeds[6][x_e + y_s*params.nx]; /* north-west */
  }

  return tot_u;
}

float timestep(const t_param params, const ArrayGrid *restrict cells, const ArrayGrid *restrict tmp_cells, int *restrict obstacles, const obstacle_loc_array *restrict obst_locs)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float two_c_four = c_sq * c_sq * 2.0f;
  const float two_c_sq = c_sq * 2.0f;
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  float tot_u;          /* accumulated magnitudes of velocity for each cell */
  tot_u = 0.f;
  
  const float accel_w1 = params.density * params.accel / 9.f;
  const float accel_w2 = params.density * params.accel / 36.f;


  // __assume_aligned(cells->speeds, 64);

  // __assume_aligned(obstacles, 64);
  // __assume_aligned(cells->neighbours, 64);
  // __assume_aligned(tmp_cells->neighbours, 64);
  //__assume_aligned(tmp_cells->speeds, 64);
  //__assume_aligned(cells->speeds, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.ny) % 2 == 0);

  /* modify the 2nd row of the grid */
  const int jj = params.ny - 2;

  #pragma omp simd
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    const int cell_i = (ii + jj * params.nx) * 9;
    if (!obstacles[ii + jj*params.nx]
        && (cells->speeds[3 + cell_i] - accel_w1) > 0.f
        && (cells->speeds[6 + cell_i] - accel_w2) > 0.f
        && (cells->speeds[7 + cell_i] - accel_w2) > 0.f)
    {
      
      /* increase 'east-side' densities */
      cells->speeds[1 + cell_i] += accel_w1;
      cells->speeds[5 + cell_i] += accel_w2;
      cells->speeds[8 + cell_i] += accel_w2;
      /* decrease 'west-side' densities */
      cells->speeds[3 + cell_i] -= accel_w1;
      cells->speeds[6 + cell_i] -= accel_w2;
      cells->speeds[7 + cell_i] -= accel_w2;
    }
  }

  /* loop over _all_ cells */
  #pragma omp parallel for reduction(+:tot_u) shared(tmp_cells, cells)
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      const int cell_i = (ii + jj * params.nx) * 9;
      const int cell_i_n = (ii + jj * params.nx) * 4;
      const int y_n = cells->neighbours[0 + cell_i_n];
      const int x_e = cells->neighbours[1 + cell_i_n];
      const int y_s = cells->neighbours[2 + cell_i_n];
      const int x_w = cells->neighbours[3 + cell_i_n];

      tmp_cells->speeds[cell_i + 0] = cells->speeds[(ii + jj*params.nx) * 9 + 0]; /* central cell, no movement */ // TODO: Useless copy
      tmp_cells->speeds[cell_i + 1] = cells->speeds[x_w + (jj*params.nx * 9) + 1]; /* east */
      tmp_cells->speeds[cell_i + 2] = cells->speeds[(ii + y_s*params.nx) * 9 + 2]; /* north */
      tmp_cells->speeds[cell_i + 3] = cells->speeds[x_e + (jj*params.nx * 9) + 3]; /* west */
      tmp_cells->speeds[cell_i + 4] = cells->speeds[(ii + y_n*params.nx) * 9 + 4]; /* south */
      tmp_cells->speeds[cell_i + 5] = cells->speeds[x_w + (y_s*params.nx * 9) + 5]; /* north-east */
      tmp_cells->speeds[cell_i + 6] = cells->speeds[x_e + (y_s*params.nx * 9) + 6]; /* north-west */
      tmp_cells->speeds[cell_i + 7] = cells->speeds[x_e + (y_n*params.nx * 9) + 7]; /* south-west */
      tmp_cells->speeds[cell_i + 8] = cells->speeds[x_w + (y_n*params.nx * 9) + 8]; /* south-east */

      /* compute local density total */
      float local_density = 0.f;

      #pragma omp simd
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        local_density += tmp_cells->speeds[cell_i + kk];
      }

      /* compute x velocity component */
      const float u_x = (tmp_cells->speeds[cell_i + 1]
                  + tmp_cells->speeds[cell_i + 5]
                  + tmp_cells->speeds[cell_i + 8]
                - (tmp_cells->speeds[cell_i + 3]
                  + tmp_cells->speeds[cell_i + 6]
                  + tmp_cells->speeds[cell_i + 7]))
                  / local_density;
      /* compute y velocity component */
      const float u_y = (tmp_cells->speeds[cell_i + 2]
                  + tmp_cells->speeds[cell_i + 5]
                  + tmp_cells->speeds[cell_i + 6]
                - (tmp_cells->speeds[cell_i + 4]
                  + tmp_cells->speeds[cell_i + 7]
                  + tmp_cells->speeds[cell_i + 8]))
                  / local_density;

      /* velocity squared */
      const float u_sq_over_two_c_sq = (u_x * u_x + u_y * u_y) / two_c_sq;

      /* directional velocity components */
      float u[NSPEEDS];
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

      d_equ[0] = w0 * local_density * (1.f - u_sq_over_two_c_sq);

      /* axis speeds: weight w1 */
      d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                        + (u[1] * u[1]) / (two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                        + (u[2] * u[2]) / (two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                        + (u[3] * u[3]) / (two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                        + (u[4] * u[4]) / (two_c_four)
                                        - u_sq_over_two_c_sq);
      /* diagonal speeds: weight w2 */
      d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                        + (u[5] * u[5]) / (two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                        + (u[6] * u[6]) / (two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                        + (u[7] * u[7]) / (two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                        + (u[8] * u[8]) / (two_c_four)
                                        - u_sq_over_two_c_sq);

      /* relaxation step */
      #pragma omp simd
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        tmp_cells->speeds[cell_i + kk] += params.omega * (d_equ[kk] - tmp_cells->speeds[cell_i + kk]);
      }

      tot_u = tot_u + (sqrtf((u_x * u_x) + (u_y * u_y)) * !(obstacles[ii + jj*params.nx]));
    }
  }

  #pragma omp parallel for shared(tmp_cells, cells)
  for (int ll = 0; ll < obst_locs->size; ll++) {
    const int obst_ptr = obst_locs->cell_ptrs[ll];
    const int ii = obst_ptr % params.nx;
    const int jj = obst_ptr / params.nx;

    const int cell_i = obst_ptr * 9;
    const int cell_i_n = obst_ptr * 4;
    const int y_n = cells->neighbours[0 + cell_i_n];
    const int x_e = cells->neighbours[1 + cell_i_n];
    const int y_s = cells->neighbours[2 + cell_i_n];
    const int x_w = cells->neighbours[3 + cell_i_n];

    tmp_cells->speeds[cell_i + 0] = cells->speeds[(jj*params.nx * 9)];
    tmp_cells->speeds[cell_i + 1] = cells->speeds[x_e + (jj*params.nx * 9) + 3]; /* west */
    tmp_cells->speeds[cell_i + 2] = cells->speeds[(ii + y_n*params.nx) * 9 + 4]; /* south */
    tmp_cells->speeds[cell_i + 3] = cells->speeds[x_w + (jj*params.nx * 9) + 1]; /* east */
    tmp_cells->speeds[cell_i + 4] = cells->speeds[(ii + y_s*params.nx) * 9 + 2]; /* north */
    tmp_cells->speeds[cell_i + 5] = cells->speeds[x_e + (y_n*params.nx * 9) + 7]; /* south-west */
    tmp_cells->speeds[cell_i + 6] = cells->speeds[x_w + (y_n*params.nx * 9) + 8]; /* south-east */
    tmp_cells->speeds[cell_i + 7] = cells->speeds[x_w + (y_s*params.nx * 9) + 5]; /* north-east */
    tmp_cells->speeds[cell_i + 8] = cells->speeds[x_e + (y_s*params.nx * 9) + 6]; /* north-west */
  }
  return tot_u;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

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

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[ii + jj*params.nx].speeds[1]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[8]
                      - (cells[ii + jj*params.nx].speeds[3]
                         + cells[ii + jj*params.nx].speeds[6]
                         + cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[6]
                      - (cells[ii + jj*params.nx].speeds[4]
                         + cells[ii + jj*params.nx].speeds[7]
                         + cells[ii + jj*params.nx].speeds[8]))
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

void init_grid(SoAGrid*restrict* grid_ptr, t_param* params) {
  (*grid_ptr) = (SoAGrid*)malloc(sizeof(SoAGrid));
  for (int i = 0; i < 9; i++) {
    (**grid_ptr).speeds[i] = (float*)_mm_malloc(sizeof(float) * params->ny * params->nx, 64);
  }
  for (int i = 0; i < 4; i++) {
    (**grid_ptr).neighbours[i] = (int*)_mm_malloc(sizeof(int) * params->ny * params->nx, 64);
  }
  (**grid_ptr).obstacle = (char*)_mm_malloc(sizeof(char) * params->ny * params->nx, 64);
}

void free_grid(SoAGrid*restrict* grid_ptr) {
  _mm_free((**grid_ptr).obstacle);
  for (int i = 0; i < 4; i++) {
    _mm_free((**grid_ptr).neighbours[i]);
  }
  for (int i = 0; i < 9; i++) {
    _mm_free((**grid_ptr).speeds[i]);
  }
  free(*grid_ptr);
}

void init_array_grid(ArrayGrid*restrict* grid_ptr, t_param* params) {
  (*grid_ptr) = (ArrayGrid*)malloc(sizeof(ArrayGrid));
  (*grid_ptr)->speeds = _mm_malloc(sizeof(float) * params->nx * params->ny * 9, 64);
  (*grid_ptr)->neighbours = _mm_malloc(sizeof(int) * params->nx * params->ny * 4, 64);
}

void free_array_grid(ArrayGrid*restrict* grid_ptr) {
  _mm_free((**grid_ptr).neighbours);
  _mm_free((**grid_ptr).speeds);
  free(*grid_ptr);
}

int initialise(const char* paramfile,
               const char* obstaclefile,
               t_param* params, 
               ArrayGrid*restrict* cells_ptr, 
               ArrayGrid*restrict* tmp_cells_ptr,
               int*restrict* obstacles_ptr, 
               float** av_vels_ptr, 
               SoAGrid*restrict* grid_ptr, 
               SoAGrid*restrict* tmp_grid_ptr, 
               t_speed*restrict* aos_grid_ptr, 
               t_speed*restrict* tmp_aos_grid_ptr,
               obstacle_loc_array*restrict* obst_locs,
               float* tot_cells)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  init_array_grid(cells_ptr, params);
  init_array_grid(tmp_cells_ptr, params);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  *obst_locs = (obstacle_loc_array*)malloc(sizeof(obstacle_loc_array));
  (*obst_locs)->cell_ptrs = (int*)malloc(sizeof(int) * params->ny * params->nx);
  *aos_grid_ptr = (t_speed*)_mm_malloc(sizeof(t_speed) * params->ny * params->nx, 64);
  *tmp_aos_grid_ptr = (t_speed*)_mm_malloc(sizeof(t_speed) * params->ny * params->nx, 64);

  init_grid(grid_ptr, params);
  init_grid(tmp_grid_ptr, params);


   /* first set all cells in obstacle array to zero */
  #pragma omp parallel for
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
      (**grid_ptr).obstacle[ii + jj * params->nx] = 0;
      (**tmp_grid_ptr).obstacle[ii + jj * params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  (*obst_locs)->size = 0;
  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
    (**grid_ptr).obstacle[xx + yy*params->nx] = blocked;
    (**tmp_grid_ptr).obstacle[xx + yy*params->nx] = blocked;
    (*obst_locs)->cell_ptrs[(*obst_locs)->size] = xx + (yy * params->nx);
    (*obst_locs)->size++;
  }

  /* and close the file */
  fclose(fp);


  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;
  #pragma omp parallel for schedule(static) shared(aos_grid_ptr, tmp_aos_grid_ptr, grid_ptr, tmp_grid_ptr, cells_ptr, tmp_cells_ptr)
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      int cell_i = ii + jj * params->nx;
      int cell_i_array = (ii + jj * params->nx) * 9;
      int cell_i_array_n = (ii + jj * params->nx) * 4;

      int y_n = (jj + 1) % params->ny;
      int x_e = (ii + 1) % params->nx;
      int y_s = (jj == 0) ? (jj + params->ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params->nx - 1) : (ii - 1);

      (*aos_grid_ptr)[cell_i].speeds[0] = w0;
      (*aos_grid_ptr)[cell_i].speeds[1] = w1;
      (*aos_grid_ptr)[cell_i].speeds[2] = w1;
      (*aos_grid_ptr)[cell_i].speeds[3] = w1;
      (*aos_grid_ptr)[cell_i].speeds[4] = w1;
      (*aos_grid_ptr)[cell_i].speeds[5] = w2;
      (*aos_grid_ptr)[cell_i].speeds[6] = w2;
      (*aos_grid_ptr)[cell_i].speeds[7] = w2; 
      (*aos_grid_ptr)[cell_i].speeds[8] = w2;

      (*aos_grid_ptr)[cell_i].neighbours[0] = y_n;
      (*aos_grid_ptr)[cell_i].neighbours[1] = x_e;
      (*aos_grid_ptr)[cell_i].neighbours[2] = y_s;
      (*aos_grid_ptr)[cell_i].neighbours[3] = x_w;

      (*tmp_aos_grid_ptr)[cell_i].neighbours[0] = y_n;
      (*tmp_aos_grid_ptr)[cell_i].neighbours[1] = x_e;
      (*tmp_aos_grid_ptr)[cell_i].neighbours[2] = y_s;
      (*tmp_aos_grid_ptr)[cell_i].neighbours[3] = x_w;

      /* centre */
      (*cells_ptr)->speeds[cell_i_array + 0] = w0;
      /* axis directions */
      (*cells_ptr)->speeds[cell_i_array + 1] = w1;
      (*cells_ptr)->speeds[cell_i_array + 2] = w1;
      (*cells_ptr)->speeds[cell_i_array + 3] = w1;
      (*cells_ptr)->speeds[cell_i_array + 4] = w1;
      /* diagonals */
      (*cells_ptr)->speeds[cell_i_array + 5] = w2;
      (*cells_ptr)->speeds[cell_i_array + 6] = w2;
      (*cells_ptr)->speeds[cell_i_array + 7] = w2;
      (*cells_ptr)->speeds[cell_i_array + 8] = w2;
      /* neighbors */
      (*cells_ptr)->neighbours[cell_i_array_n + 0] = y_n;
      (*cells_ptr)->neighbours[cell_i_array_n + 1] = x_e * 9;
      (*cells_ptr)->neighbours[cell_i_array_n + 2] = y_s;
      (*cells_ptr)->neighbours[cell_i_array_n + 3] = x_w * 9;

      (*tmp_cells_ptr)->neighbours[cell_i_array_n + 0] = y_n;
      (*tmp_cells_ptr)->neighbours[cell_i_array_n + 1] = x_e * 9;
      (*tmp_cells_ptr)->neighbours[cell_i_array_n + 2] = y_s;
      (*tmp_cells_ptr)->neighbours[cell_i_array_n + 3] = x_w * 9;

      /* Structure of Arrays */

      /* Speeds */
      (**grid_ptr).speeds[0][cell_i] = w0;
      (**grid_ptr).speeds[1][cell_i] = w1;
      (**grid_ptr).speeds[2][cell_i] = w1;
      (**grid_ptr).speeds[3][cell_i] = w1;
      (**grid_ptr).speeds[4][cell_i] = w1;
      (**grid_ptr).speeds[5][cell_i] = w2;
      (**grid_ptr).speeds[6][cell_i] = w2;
      (**grid_ptr).speeds[7][cell_i] = w2;
      (**grid_ptr).speeds[8][cell_i] = w2;

      /* Neighbors */
      (**grid_ptr).neighbours[0][cell_i] = y_n;
      (**grid_ptr).neighbours[1][cell_i] = x_e;
      (**grid_ptr).neighbours[2][cell_i] = y_s;
      (**grid_ptr).neighbours[3][cell_i] = x_w;

      (**tmp_grid_ptr).neighbours[0][cell_i] = y_n;
      (**tmp_grid_ptr).neighbours[1][cell_i] = x_e;
      (**tmp_grid_ptr).neighbours[2][cell_i] = y_s;
      (**tmp_grid_ptr).neighbours[3][cell_i] = x_w;
    }
  }

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  *tot_cells = params->nx * params->ny - (*obst_locs)->size; 
  return EXIT_SUCCESS;
}

int finalise(const t_param* params, ArrayGrid*restrict* cells_ptr, ArrayGrid*restrict* tmp_cells_ptr,
             int*restrict* obstacles_ptr, float** av_vels_ptr, SoAGrid*restrict* grid_ptr, SoAGrid*restrict* tmp_grid_ptr, t_speed*restrict* aos_grid_ptr, t_speed*restrict* tmp_aos_grid_ptr, obstacle_loc_array*restrict* obst_locs)
{
  /*
  ** free up allocated memory
  */
  free(*obst_locs);
  *obst_locs = NULL;
  
  _mm_free(*aos_grid_ptr);
  *aos_grid_ptr = NULL;

  _mm_free(*tmp_aos_grid_ptr);
  *tmp_aos_grid_ptr = NULL;

  free_array_grid(cells_ptr);
  *cells_ptr = NULL;

  free_array_grid(tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  free_grid(grid_ptr);
  grid_ptr = NULL;

  free_grid(tmp_grid_ptr);
  tmp_grid_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float final_av_vel)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return final_av_vel * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, ArrayGrid* cells, int* obstacles, float* av_vels, int soa, SoAGrid* grid, t_speed* aos_grid_ptr)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      float speeds[NSPEEDS];
      int obstacle;
      if (soa == 2)
      {
        int cell_i = (ii + jj * params.nx);
        for (int ss = 0; ss < NSPEEDS; ss++) 
        {
          speeds[ss] = aos_grid_ptr[cell_i].speeds[ss];
        }
        obstacle = obstacles[cell_i];
      }
      else if (soa == 0)
      {
        int cell_i = (ii + jj * params.nx) * 9;
        for (int ss = 0; ss < NSPEEDS; ss++)
        {
          speeds[ss] = cells->speeds[cell_i + ss];
        }
        obstacle = obstacles[ii + jj * params.nx];
      }
      else 
      {
        int cell_i = (ii + jj * params.nx);
        for (int ss = 0; ss < NSPEEDS; ss++) 
        {
          speeds[ss] = grid->speeds[ss][cell_i];
        }
        obstacle = grid->obstacle[cell_i];
      }

      if (obstacle)
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += speeds[kk];
        }

        /* compute x velocity component */
        u_x = (speeds[1]
              + speeds[5]
              + speeds[8]
              - (speeds[3]
                  + speeds[6]
                  + speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (speeds[2]
              + speeds[5]
              + speeds[6]
              - (speeds[4]
                  + speeds[7]
                  + speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacle);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
