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
#include <mpi.h>

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

typedef struct
{
  int nprocs;
  int rank;
  int work;
  int start;
  int end;
  int up;
  int down;
  int row_length;
  int spillover;
} rank_params;

typedef struct
{
  float c_sq;
  float two_c_four;
  float two_c_sq;
  float w0;
  float w1;
  float w2;
} physical_consts;

/*
** function prototypes
*/

void read_params(const char* paramfile, t_param* params);

/* allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* obstaclefile,
               const t_param params,
               ArrayGrid*restrict* cells_ptr,
               ArrayGrid*restrict* tmp_cells_ptr,
               ArrayGrid*restrict* global_grid,
               int*restrict* obstacles_ptr,
               float** av_vels_ptr,
               float** accum_av_vels_ptr,
               SoAGrid*restrict* grid_ptr,
               SoAGrid*restrict* tmp_grid_ptr,
               obstacle_loc_array*restrict* obst_locs,
               float* tot_cells,
               const rank_params* r_params);


/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, const ArrayGrid *restrict cells, const ArrayGrid *restrict tmp_cells, int *restrict obstacles, const obstacle_loc_array*restrict obst_locs, const rank_params* r_params, const physical_consts* p_cs);
float timestep_SoA(const t_param params, const SoAGrid *restrict grid, const SoAGrid *restrict tmp_grid, const obstacle_loc_array *restrict obst_locs, const int *restrict obstacles);
int write_values(const t_param params, ArrayGrid* cells, int* obstacles, float* av_vels, int soa, SoAGrid* grid, char* filename);
int read_obstacles(const char* obstacle_file, const t_param params, int*restrict* obstacles_ptr, obstacle_loc_array*restrict* obst_locs, const int work, const int start, const int end);

void halo_exchange(const t_param params, const ArrayGrid *restrict cells, const rank_params* r_params);
void recieve_rows(const t_param params, const ArrayGrid *restrict cells, const rank_params* r_params);
void send_rows(const t_param params, const ArrayGrid *restrict cells, const rank_params* r_params);
void collate_rank_data(const t_param params, float *av_vels, float *accum_av_vels, const ArrayGrid *restrict cells, ArrayGrid *global_grid, const rank_params* r_params);


/* finalise, including freeing up allocated memory */
int free_obstacles(int*restrict* obstacles_ptr, obstacle_loc_array*restrict* obst_locs);
int finalise(const t_param* params,
             ArrayGrid*restrict* cells_ptr,
             ArrayGrid*restrict* tmp_cells_ptr, 
             ArrayGrid*restrict* global_grid,
             int*restrict* obstacles_ptr, 
             float** av_vels_ptr, 
             float** accum_av_vels_ptr, 
             SoAGrid*restrict* grid_ptr, 
             SoAGrid*restrict* tmp_grid_ptr, 
             obstacle_loc_array*restrict* obst_locs);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float final_av_vel);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

void display_obstacle_grid(const t_param params, int *restrict obstacles, const int work) {
  for (int jj = 0; jj < work; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      printf(" %d ", obstacles[ii + jj * params.nx]);
    }
    printf("\n");
  }
  return;
}

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  char*    paramfile            = NULL;    /* name of the input parameter file */
  char*    obstaclefile         = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  ArrayGrid *restrict cells     = NULL;    /* grid containing fluid densities */
  ArrayGrid *restrict tmp_cells = NULL;    /* scratch space */
  ArrayGrid *restrict global_grid = NULL;
  int       *restrict obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels                = NULL;     /* a record of the av. velocity computed for each timestep */
  float* accum_av_vels          = NULL;
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */
  SoAGrid* grid     = NULL;
  SoAGrid* tmp_grid = NULL;
  obstacle_loc_array *restrict obst_locs = NULL;
  float tot_cells = 0;
  char SoA = 0; // 0 = False
  char render = 1;
  rank_params *r_params = malloc(sizeof(rank_params));
  physical_consts *p_consts = malloc(sizeof(physical_consts));
  
  p_consts->c_sq = 1.f / 3.f; /* square of speed of sound */
  p_consts->two_c_four = p_consts->c_sq * p_consts->c_sq * 2.0f;
  p_consts->two_c_sq = p_consts->c_sq * 2.0f;
  p_consts->w0 = 4.f / 9.f;  /* weighting factor */
  p_consts->w1 = 1.f / 9.f;  /* weighting factor */
  p_consts->w2 = 1.f / 36.f; /* weighting factor */

#ifdef SOA
  SoA = 1;
#endif

  /* parse the command line */
  if (argc != 3)
  {
    // usage(argv[0]);
    paramfile = "inputs/input_1024x1024.params";
    obstaclefile = "obstacles/obstacles_1024x1024.dat"; 
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  // Get rank info
  MPI_Comm_size(MPI_COMM_WORLD, &r_params->nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &r_params->rank);

  // Extract input parameters from paramfile
  read_params(paramfile, &params);

  // Work bounds
  r_params->spillover = params.ny % r_params->nprocs;
  r_params->work = (params.ny / r_params->nprocs) + (1 * r_params->rank < r_params->spillover);

  r_params->start = r_params->rank * r_params->work + (r_params->spillover * (r_params->rank >= r_params->spillover));
  r_params->end = r_params->start + r_params->work;

  r_params->row_length = params.nx * 9;
  r_params->up = r_params->rank == 0 ? (r_params->nprocs - 1) : (r_params->rank - 1);
  r_params->down = r_params->rank == (r_params->nprocs - 1) ? 0 : (r_params->rank + 1);

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(obstaclefile, params, &cells, &tmp_cells, &global_grid, &obstacles, &av_vels, &accum_av_vels, &grid, &tmp_grid, &obst_locs, &tot_cells, r_params);
  
  // Initialise halo regions
  send_rows(params, cells, r_params);
  recieve_rows(params, cells, r_params);  

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  if (SoA == 0) {
    for (int tt = 0; tt < params.maxIters; tt++)
    {
      av_vels[tt] = timestep(params, cells, tmp_cells, obstacles, obst_locs, r_params, p_consts);
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
   
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;
  // Collate data from ranks here 

  collate_rank_data(params, av_vels, accum_av_vels, cells, global_grid, r_params);
  /* Total/collate time stops here.*/

  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  /* write final values and free memory */
  if (r_params->rank == 0) {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, av_vels[params.maxIters - 1]));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  }
  
  if (r_params->rank == 0) {
    for (int i = 0; i < params.maxIters; i++) {
      accum_av_vels[i] = accum_av_vels[i] / tot_cells;
    }

    int *restrict global_obstacles = NULL;
    obstacle_loc_array *restrict global_obst_locs = NULL;
    read_obstacles(obstaclefile, params, &global_obstacles, &global_obst_locs, params.ny, 0, params.ny);
    write_values(params, global_grid, global_obstacles, accum_av_vels, SoA, grid, "final_state.dat");
    free_obstacles(&global_obstacles, &global_obst_locs);
  }

  free(p_consts);
  finalise(&params, &cells, &tmp_cells, &global_grid, &obstacles, &av_vels, &accum_av_vels, &grid, &tmp_grid, &obst_locs);
  MPI_Finalize();
  return EXIT_SUCCESS;
}

void collate_rank_data(const t_param params, float *av_vels, float *accum_av_vels, const ArrayGrid *restrict cells, ArrayGrid *global_grid, const rank_params* r_params) {
  int recv_counts[r_params->nprocs];
  int displs[r_params->nprocs];
  for (int i = 0; i < r_params->nprocs; i++) {
    recv_counts[i] = (((params.ny / r_params->nprocs) + (1 * i < r_params->spillover))) * r_params->row_length;
  }

  if (r_params->rank == 0) {
    displs[0] = 0;
    for (int i = 1; i < r_params->nprocs; i++) {
      displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
  }

  MPI_Gatherv(
    &(cells->speeds[r_params->row_length * 1]),
    r_params->work * r_params->row_length,
    MPI_FLOAT,
    &(global_grid->speeds[0]),
    recv_counts,
    displs,
    MPI_FLOAT,
    0,
    MPI_COMM_WORLD
  );

  MPI_Reduce(av_vels, accum_av_vels, params.maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
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

void halo_exchange(const t_param params, const ArrayGrid *restrict cells, const rank_params* r_params) {
  MPI_Status status;

  MPI_Sendrecv(
    &cells->speeds[1 * r_params->row_length],                    // Send buffer
    r_params->row_length,                                        // Row length
    MPI_FLOAT,                                                   // Send datatype
    r_params->up,                                                // Destination rank
    1,                                                           // Send tag
    &cells->speeds[(r_params->work + 1) * r_params->row_length], // Recieve buffer
    r_params->row_length,                                        // Row length
    MPI_FLOAT,                                                   // Recieve datatype
    r_params->down,                                              // Source rank
    1,                                                           // Recieve tag
    MPI_COMM_WORLD,                                              // Communicator handle
    &status                                                      // Status container
  );

  // Send down, recieve up
  MPI_Sendrecv(
    &cells->speeds[r_params->work * r_params->row_length],
    r_params->row_length,
    MPI_FLOAT,
    r_params->down,
    1,
    &cells->speeds[0],
    r_params->row_length,
    MPI_FLOAT,
    r_params->up,
    1,
    MPI_COMM_WORLD,
    &status
  );
}

void send_rows(const t_param params, const ArrayGrid *restrict cells, const rank_params* r_params) {
  // Non-blocking row sends
  
  // Send to up
  MPI_Request up_request;
  MPI_Isend(
    &cells->speeds[1 * r_params->row_length],
    r_params->row_length,
    MPI_FLOAT,
    r_params->up,
    1,
    MPI_COMM_WORLD,
    &up_request
  );

  // Send to down
  MPI_Request down_request;
  MPI_Isend(
    &cells->speeds[r_params->work * r_params->row_length],
    r_params->row_length,
    MPI_FLOAT,
    r_params->down,
    1,
    MPI_COMM_WORLD,
    &down_request
  );
}

void recieve_rows(const t_param params, const ArrayGrid *restrict cells, const rank_params* r_params) {
  // Blocking row recieves
  
  // Recieve from down
  MPI_Status down_status;
  MPI_Recv(
    &cells->speeds[(r_params->work + 1) * r_params->row_length],
    r_params->row_length,
    MPI_FLOAT,
    r_params->down,
    1,
    MPI_COMM_WORLD,
    &down_status
  );

  // Recieve from up
  MPI_Status up_status; 
  MPI_Recv(
    &cells->speeds[0],
    r_params->row_length,
    MPI_FLOAT,
    r_params->up,
    1,
    MPI_COMM_WORLD,
    &up_status
  );
}

float process_row(const t_param params, int jj, const physical_consts *p_cs, const ArrayGrid *restrict cells, const ArrayGrid *restrict tmp_cells, int *restrict obstacles) {
  float tot_u = 0;
  #pragma omp simd
  for (int ii = 0; ii < params.nx; ii++)
  {
    const int cell_i = (ii + jj * params.nx) * 9;
    const int cell_i_n = (ii + (jj - 1) * params.nx) * 4;
    const int y_n = cells->neighbours[0 + cell_i_n];
    const int x_e = cells->neighbours[1 + cell_i_n];
    const int y_s = cells->neighbours[2 + cell_i_n];
    const int x_w = cells->neighbours[3 + cell_i_n];

    tmp_cells->speeds[cell_i + 0] = cells->speeds[(ii + jj*params.nx) * 9 + 0]; /* central cell, no movement */
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
      const float u_sq_over_two_c_sq = (u_x * u_x + u_y * u_y) / p_cs->two_c_sq;

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

      d_equ[0] = p_cs->w0 * local_density * (1.f - u_sq_over_two_c_sq);

      /* axis speeds: weight w1 */
      d_equ[1] = p_cs->w1 * local_density * (1.f + u[1] / p_cs->c_sq
                                        + (u[1] * u[1]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[2] = p_cs->w1 * local_density * (1.f + u[2] / p_cs->c_sq
                                        + (u[2] * u[2]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[3] = p_cs->w1 * local_density * (1.f + u[3] / p_cs->c_sq
                                        + (u[3] * u[3]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[4] = p_cs->w1 * local_density * (1.f + u[4] / p_cs->c_sq
                                        + (u[4] * u[4]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);
      /* diagonal speeds: weight w2 */
      d_equ[5] = p_cs->w2 * local_density * (1.f + u[5] / p_cs->c_sq
                                        + (u[5] * u[5]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[6] = p_cs->w2 * local_density * (1.f + u[6] / p_cs->c_sq
                                        + (u[6] * u[6]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[7] = p_cs->w2 * local_density * (1.f + u[7] / p_cs->c_sq
                                        + (u[7] * u[7]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[8] = p_cs->w2 * local_density * (1.f + u[8] / p_cs->c_sq
                                        + (u[8] * u[8]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);

    /* relaxation step */
    #pragma omp simd
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      tmp_cells->speeds[cell_i + kk] += params.omega * (d_equ[kk] - tmp_cells->speeds[cell_i + kk]);
    }

    tot_u = tot_u + (sqrtf((u_x * u_x) + (u_y * u_y)) * !(obstacles[ii + (jj - 1)*params.nx]));
  }
  return tot_u;
}

float timestep(const t_param params, const ArrayGrid *restrict cells, const ArrayGrid *restrict tmp_cells, int *restrict obstacles, const obstacle_loc_array *restrict obst_locs, const rank_params* r_params, const physical_consts* p_cs)
{
  send_rows(params, cells, r_params);

  float tot_u = 0;
  
  const float accel_w1 = params.density * params.accel / 9.f;
  const float accel_w2 = params.density * params.accel / 36.f;

  __assume_aligned(cells->speeds, 64);

  __assume_aligned(obstacles, 64);
  __assume_aligned(cells->neighbours, 64);
  __assume_aligned(tmp_cells->neighbours, 64);
  __assume_aligned(tmp_cells->speeds, 64);
  __assume_aligned(cells->speeds, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.ny) % 2 == 0);

  /* modify the 2nd row of the grid */
  const int jj = params.ny - 2;
  if (jj >= r_params->start && jj < r_params->start + r_params->work) {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell is not occupied and
      ** we don't send a negative density */
      const int cell_i = (ii + ((jj - r_params->start) + 1) * params.nx) * 9;
      if (!obstacles[ii + (jj - r_params->start)*params.nx]
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
  }

  // 0th row = rank above's row
  // 1st row = processed after 0th row is recieved
  for (int jj = 2; jj < r_params->work; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      const int cell_i = (ii + jj * params.nx) * 9;
      const int cell_i_n = (ii + (jj - 1) * params.nx) * 4;
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
      const float u_sq_over_two_c_sq = (u_x * u_x + u_y * u_y) / p_cs->two_c_sq;

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

      d_equ[0] = p_cs->w0 * local_density * (1.f - u_sq_over_two_c_sq);

      /* axis speeds: weight w1 */
      d_equ[1] = p_cs->w1 * local_density * (1.f + u[1] / p_cs->c_sq
                                        + (u[1] * u[1]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[2] = p_cs->w1 * local_density * (1.f + u[2] / p_cs->c_sq
                                        + (u[2] * u[2]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[3] = p_cs->w1 * local_density * (1.f + u[3] / p_cs->c_sq
                                        + (u[3] * u[3]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[4] = p_cs->w1 * local_density * (1.f + u[4] / p_cs->c_sq
                                        + (u[4] * u[4]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);
      /* diagonal speeds: weight w2 */
      d_equ[5] = p_cs->w2 * local_density * (1.f + u[5] / p_cs->c_sq
                                        + (u[5] * u[5]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[6] = p_cs->w2 * local_density * (1.f + u[6] / p_cs->c_sq
                                        + (u[6] * u[6]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[7] = p_cs->w2 * local_density * (1.f + u[7] / p_cs->c_sq
                                        + (u[7] * u[7]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);
      d_equ[8] = p_cs->w2 * local_density * (1.f + u[8] / p_cs->c_sq
                                        + (u[8] * u[8]) / (p_cs->two_c_four)
                                        - u_sq_over_two_c_sq);

      /* relaxation step */
      #pragma omp simd
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        tmp_cells->speeds[cell_i + kk] += params.omega * (d_equ[kk] - tmp_cells->speeds[cell_i + kk]);
      }

      tot_u = tot_u + (sqrtf((u_x * u_x) + (u_y * u_y)) * !(obstacles[ii + (jj - 1)*params.nx]));
    }
  }
  recieve_rows(params, cells, r_params);

  tot_u += process_row(params, 1, p_cs, cells, tmp_cells, obstacles);
  tot_u += process_row(params, r_params->work, p_cs, cells, tmp_cells, obstacles);

  #pragma omp simd
  for (int ll = 0; ll < obst_locs->size; ll++) {
    const int obst_ptr = obst_locs->cell_ptrs[ll];
    const int ii = obst_ptr % params.nx;
    const int jj = obst_ptr / params.nx;

    const int cell_i = (obst_ptr + params.nx) * 9;
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

void init_array_grid(ArrayGrid*restrict* grid_ptr, t_param params, int work) {
  (*grid_ptr) = (ArrayGrid*)malloc(sizeof(ArrayGrid));
  (*grid_ptr)->speeds = _mm_malloc(sizeof(float) * params.nx * (work + 2) * 9, 64);
  (*grid_ptr)->neighbours = _mm_malloc(sizeof(int) * params.nx * work * 4, 64);
}

void free_array_grid(ArrayGrid*restrict* grid_ptr) {
  _mm_free((**grid_ptr).neighbours);
  _mm_free((**grid_ptr).speeds);
  free(*grid_ptr);
}

void read_params(const char* paramfile, t_param* params)
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
}

int read_obstacles(const char* obstacle_file, const t_param params, int*restrict* obstacles_ptr, obstacle_loc_array*restrict* obst_locs, const int work, const int start, const int end) {
  *obstacles_ptr = malloc(sizeof(int) * work * params.nx);
  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  *obst_locs = (obstacle_loc_array*)malloc(sizeof(obstacle_loc_array));
  (*obst_locs)->cell_ptrs = (int*)malloc(sizeof(int) * work * params.nx);

  for (int jj = 0; jj < work; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      (*obstacles_ptr)[ii + jj * params.nx] = 0;
    }
  }

  FILE* fp = fopen(obstacle_file, "r");
  char message[1024];
  int retval, blocked, xx, yy;
  if (fp == NULL) {
    sprintf(message, "could not open input obstacles file: %s", obstacle_file);
    die(message, __LINE__, __FILE__);
  }

  (*obst_locs)->size = 0;
  int obst_count = 0;
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
    if (xx < 0 || xx > params.nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
    if (yy < 0 || yy > params.ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    if (yy >= start && yy < end) {
      (*obstacles_ptr)[xx + ((yy - start) * params.nx)] = blocked;
      (*obst_locs)->cell_ptrs[(*obst_locs)->size] = xx + ((yy - start) * params.nx);
      (*obst_locs)->size++; 
    }
    obst_count++;
  }
  fclose(fp);
  return obst_count;
}

int initialise(const char* obstaclefile,
               const t_param params, 
               ArrayGrid*restrict* cells_ptr, 
               ArrayGrid*restrict* tmp_cells_ptr,
               ArrayGrid*restrict* global_grid,
               int*restrict* obstacles_ptr, 
               float** av_vels_ptr,
               float** accum_av_vels_ptr,
               SoAGrid*restrict* grid_ptr, 
               SoAGrid*restrict* tmp_grid_ptr,
               obstacle_loc_array*restrict* obst_locs,
               float* tot_cells,
               const rank_params* r_params)
{
  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want  to
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
  init_array_grid(cells_ptr, params, r_params->work);
  init_array_grid(tmp_cells_ptr, params, r_params->work);
  init_array_grid(global_grid, params, params.ny - 2);
  
  int obst_count = read_obstacles(obstaclefile, params, obstacles_ptr, obst_locs, r_params->work, r_params->start, r_params->end);

  /* initialise densities */
  float w0 = params.density * 4.f / 9.f;
  float w1 = params.density      / 9.f;
  float w2 = params.density      / 36.f;
  for (int jj = 0; jj < r_params->work; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int cell_i = ii + (jj + 1) * params.nx;
      int cell_i_array = (ii + (jj + 1) * params.nx) * 9;
      int cell_i_array_n = (ii + jj * params.nx) * 4;

      int y_n = jj + 2;
      int x_e = (ii + 1) % params.nx;
      int y_s = jj;
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

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
    }
  }
  

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params.maxIters);
  *accum_av_vels_ptr = (float*)malloc(sizeof(float) * params.maxIters);

  *tot_cells = params.nx * params.ny - obst_count;
  return EXIT_SUCCESS;
}

int free_obstacles(int*restrict* obstacles_ptr, obstacle_loc_array*restrict* obst_locs) {
  free(*obst_locs);
  *obst_locs = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL; 
  return EXIT_SUCCESS;
}

int finalise(const t_param* params, ArrayGrid*restrict* cells_ptr, ArrayGrid*restrict* tmp_cells_ptr, ArrayGrid*restrict* global_grid_ptr,
             int*restrict* obstacles_ptr, float** av_vels_ptr, float** accum_av_vels_ptr, SoAGrid*restrict* grid_ptr, SoAGrid*restrict* tmp_grid_ptr, obstacle_loc_array*restrict* obst_locs)
{
  /*
  ** free up allocated memory
  */
  free_obstacles(obstacles_ptr, obst_locs);  

  free_array_grid(cells_ptr);
  *cells_ptr = NULL;

  free_array_grid(tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free_array_grid(global_grid_ptr);
  *global_grid_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  free(*accum_av_vels_ptr);
  *accum_av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float final_av_vel)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return final_av_vel * params.reynolds_dim / viscosity;
}

int write_values(const t_param params, ArrayGrid* global_grid, int* obstacles, float* av_vels, int soa, SoAGrid* grid, char* filename)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(filename, "w");

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
      if (soa == 0)
      {
        int cell_i = (ii + jj * params.nx) * 9;
        for (int ss = 0; ss < NSPEEDS; ss++)
        {
          speeds[ss] = global_grid->speeds[cell_i + ss];
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
