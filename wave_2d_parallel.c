#define _XOPEN_SOURCE 600
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "argument_utils.h"

// TODO: Remove this include
#include "log.c/src/log.h"
// TASK: T1a
// Include the MPI headerfile
// BEGIN: T1a
#include <mpi.h>
// END: T1a

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Option to change numerical precision
typedef int64_t int_t;
typedef double real_t;

// Buffers for three time steps, indexed with 2 ghost points for the boundary
real_t *buffers[3] = {NULL, NULL, NULL};

// TASK: T1b
// Declare variables each MPI process will need
// BEGIN: T1b
#define U_prv(i, j) buffers[0][((i) + 1) * (N + 2) + (j) + 1]
#define U(i, j) buffers[1][((i) + 1) * (N + 2) + (j) + 1]
#define U_nxt(i, j) buffers[2][((i) + 1) * (N + 2) + (j) + 1]

int world_rank, world_size;
// Big numbers
int local_m, local_n, local_m_offset, local_n_offset;

// Small numbers
int local_cart_coords[2];
int max_cart_m, max_cart_n;
int m_processes;
int n_processes;

MPI_Comm cart_comm;
int cart_rank;

#define MPI_ROOT_RANK 0
#define IS_MPI_ROOT_RANK (world_rank == MPI_ROOT_RANK)
#define IS_MPI_LAST (world_rank == world_size - 1)

#define IS_MPI_TOPMOST (local_cart_coords[0] == 0)
#define IS_MPI_BOTTOMMOST (local_cart_coords[0] == m_processes - 1)
#define IS_MPI_LEFTMOST (local_cart_coords[1] == 0)
#define IS_MPI_RIGHTMOST (local_cart_coords[1] == n_processes - 1)

// END: T1b

// Simulation parameters: size, step count, and how often to save the state
int_t M = 256, // rows
    N = 256,   // cols
    max_iteration = 4000, snapshot_freq = 20;

// Wave equation parameters, time step is derived from the space step
const real_t c = 1.0, dx = 1.0, dy = 1.0;
real_t dt;

// Rotate the time step buffers.
void move_buffer_window(void) {
  real_t *temp = buffers[0];
  buffers[0] = buffers[1];
  buffers[1] = buffers[2];
  buffers[2] = temp;
}

// TASK: T4
// Set up our three buffers, and fill two with an initial perturbation
// and set the time step.
void domain_initialize(void) {
  // BEGIN: T4
  buffers[0] = malloc((local_m + 2) * (local_n + 2) * sizeof(real_t));
  buffers[1] = malloc((local_m + 2) * (local_n + 2) * sizeof(real_t));
  buffers[2] = malloc((local_m + 2) * (local_n + 2) * sizeof(real_t));

  for (int_t i = local_m_offset; i < local_m_offset + local_m_offset; i++) {
    for (int_t j = local_n_offset; j < local_n_offset + local_m_offset; j++) {
      // Calculate delta (radial distance) adjusted for M x N grid
      real_t delta = sqrt(((i - M / 2.0) * (i - M / 2.0)) / (real_t)M +
                          ((j - N / 2.0) * (j - N / 2.0)) / (real_t)N);

      int_t i_local = i - local_m_offset;
      int_t j_local = j - local_n_offset;
      U_prv(i_local, j_local) = U(i_local, j_local) = exp(-4.0 * delta * delta);
    }
  }

  // Set the time step for 2D case
  dt = dx * dy / (c * sqrt(dx * dx + dy * dy));
  // END: T4
}

// Get rid of all the memory allocations
void domain_finalize(void) {
  log_debug("Rank %d: Domain finalize start", world_rank);
  free(buffers[0]);
  free(buffers[1]);
  free(buffers[2]);
}

// TASK: T5
// Integration formula
void time_step(void) {
  // BEGIN: T5
  for (int_t i = 0; i < local_m; i++) {
    for (int_t j = 0; j < local_n; j++) {
      U_nxt(i, j) = -U_prv(i, j) + 2.0 * U(i, j) +
                    (dt * dt * c * c) / (dx * dy) *
                        (U(i - 1, j) + U(i + 1, j) + U(i, j - 1) + U(i, j + 1) -
                         4.0 * U(i, j));
    }
  }
  // END: T5
}

// TASK: T6
// Communicate the border between processes.
void border_exchange(void) {
  // BEGIN: T6
  int _rank_source;
  int left_rank;
  int top_rank;
  int right_rank;
  int bottom_rank;

  MPI_Cart_shift(cart_comm, 0, 1, &top_rank, &bottom_rank);
  MPI_Cart_shift(cart_comm, 1, 1, &left_rank, &right_rank);

  // Send left borders
  // REMEMBER cord[0] == m == rows
  // REMEMBER cord[1] == n == cols
  // Not Leftmost
  if (!IS_MPI_LEFTMOST) {

    // Sending and receve could probalby be done on the same memory allocation
    real_t *send_column = malloc(local_m * sizeof(real_t));

    // Filling the column with left
    for (int_t i = 0; i < local_m; i++) {
      send_column[i] = U(i, 0);
    }

    // Send Left Column
    MPI_Send(send_column, local_m, MPI_DOUBLE, left_rank, 0, cart_comm);

    free(send_column);
  }

  log_trace("Rank: %d: Sendt left border", world_rank);

  // Not Rightmost
  if (!IS_MPI_RIGHTMOST) {
    // Recive right borders
    real_t *recv_column = malloc(local_m * sizeof(real_t));

    MPI_Recv(recv_column, local_m, MPI_DOUBLE, right_rank, 0, cart_comm,
             MPI_STATUS_IGNORE);

    // Insert into right side of U
    for (int_t i = 0; i < local_m; i++) {
      U(i, local_n) = recv_column[i];
    }
    free(recv_column);
    // Send right borders

    real_t *send_column = malloc(local_m * sizeof(real_t));

    // Filling the column with right
    for (int_t i = 0; i < local_m; i++) {
      send_column[i] = U(i, local_n - 1);
    }

    // Send Right Column
    MPI_Send(send_column, local_m, MPI_DOUBLE, right_rank, 0, cart_comm);
    free(send_column);
  }
  log_trace("Rank: %d: Recv left border", world_rank);

  // Recive left borders
  // Not Leftmost
  if (!IS_MPI_LEFTMOST) {
    real_t *recv_column = malloc(local_m * sizeof(real_t));

    MPI_Recv(recv_column, local_m, MPI_DOUBLE, left_rank, 0, cart_comm,
             MPI_STATUS_IGNORE);

    // Insert into left side of U
    for (int_t i = 0; i < local_m; i++) {
      U(i, -1) = recv_column[i];
    }
    free(recv_column);
  }

  log_trace("Rank: %d: Send right border", world_rank);

  // Send top borders
  // Not Topmost
  if (!IS_MPI_TOPMOST) {
    // Send Top Row
    MPI_Send(&U(0, 0), local_n, MPI_DOUBLE, top_rank, 0, cart_comm);
  }

  log_trace("Rank: %d: Sendt top border", world_rank);

  // Not Bottommost
  // Recive top borders, and insert into bottom borders
  if (!IS_MPI_BOTTOMMOST) {
    MPI_Recv(&U(local_m, 0), local_n, MPI_DOUBLE, bottom_rank, 0, cart_comm,
             MPI_STATUS_IGNORE);

    // Send Bottom Row
    MPI_Send(&U(local_m - 1, 0), local_n, MPI_DOUBLE, bottom_rank, 0,
             cart_comm);
  }

  if (!IS_MPI_TOPMOST) {
    // Recive bottom borders
    MPI_Recv(&U(-1, 0), local_n, MPI_DOUBLE, top_rank, 0, cart_comm,
             MPI_STATUS_IGNORE);
  }

  log_trace("Rank: %d: Recv top border", world_rank);
  // Send and receive data from the top and bottom
  // END: T6
}

// TASK: T7
// Neumann (reflective) boundary condition
void boundary_condition(void) {
  // BEGIN: T7
  for (int_t i = 0; i < M; i++) {
    if (IS_MPI_LEFTMOST)
      U(i, -1) = U(i, 1);

    if (IS_MPI_RIGHTMOST)
      U(i, N) = U(i, N - 2);
  }
  for (int_t j = 0; j < N; j++) {
    if (IS_MPI_TOPMOST)
      U(-1, j) = U(1, j);

    if (IS_MPI_BOTTOMMOST)
      U(M, j) = U(M - 2, j);
  }
  // END: T7
}

// TASK: T8
// Save the present time step in a numbered file under 'data/'
void domain_save(int_t step) {
  // BEGIN: T8
  /* char filename[256]; */
  /* sprintf(filename, "data/%.5ld.dat", step); */
  /* FILE *out = fopen(filename, "wb"); */
  /* for (int_t i = 0; i < M; i++) { */
  /*   fwrite(&U(i, 0), sizeof(real_t), N, out); */
  /* } */
  /* fclose(out); */
  // Inspiration is taken from
  // https://stackoverflow.com/questions/33537451/writing-distributed-arrays-using-mpi-io-and-cartesian-topology

  /* Create derived datatype for interior grid (output grid) */
  MPI_Datatype grid;
  int start[2] = {1, 1};
  int arrsize[2] = {local_m + 2, local_n + 2};
  int gridsize[2] = {local_m, local_n};

  log_debug("Rank %d: Creating derived datatype", world_rank);
  log_debug("Rank %d: start: (%d, %d), arrsize: (%d, %d), gridsize: (%d, %d)",
            world_rank, start[0], start[1], arrsize[0], arrsize[1], gridsize[0],
            gridsize[1]);
  MPI_Type_create_subarray(2, arrsize, gridsize, start, MPI_ORDER_C, MPI_DOUBLE,
                           &grid);
  MPI_Type_commit(&grid);

  MPI_Datatype view;

  int startV[2] = {local_m * local_cart_coords[0],
                   local_n * local_cart_coords[1]};
  int arrsizeV[2] = {M, N};
  int gridsizeV[2] = {local_m, local_n};

  log_debug("Rank %d: Creating derived datatype 2", world_rank);
  log_debug(
      "Rank %d: startV: (%d, %d), arrsizeV: (%d, %d), gridsizeV: (%d, %d)",
      world_rank, startV[0], startV[1], arrsizeV[0], arrsizeV[1], gridsizeV[0],
      gridsizeV[1]);
  MPI_Type_create_subarray(2, arrsizeV, gridsizeV, startV, MPI_ORDER_C,
                           MPI_DOUBLE, &view);
  MPI_Type_commit(&view);

  /* MPI IO */
  MPI_File fh;

  char filename[256];
  sprintf(filename, "data/%.5ld.dat", step);
  log_info("Rank %d: Writing to file: %s", world_rank, filename);

  MPI_File_open(cart_comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fh);

  MPI_File_set_view(fh, 0, MPI_DOUBLE, view, "native", MPI_INFO_NULL);
  MPI_File_write_all(fh, &U(0, 0), 1, grid, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  /* Cleanup */
  MPI_Type_free(&view);
  MPI_Type_free(&grid);
  // END: T8
}

// Main time integration.
void simulate(void) {
  // Go through each time step
  for (int_t iteration = 0; iteration <= max_iteration; iteration++) {
    if ((iteration % snapshot_freq) == 0) {
      domain_save(iteration / snapshot_freq);
    }

    // Derive step t+1 from steps t and t-1
    border_exchange();
    boundary_condition();
    time_step();

    // Rotate the time step buffers
    move_buffer_window();
  }
}

int main(int argc, char **argv) {
  // TODO: remove this
  log_set_level(LOG_DEBUG);
  // TASK: T1c
  // Initialise MPI
  // BEGIN: T1c
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // END: T1c
  //

  // TASK: T3
  // Distribute the user arguments to all the processes
  // BEGIN: T3
  OPTIONS *options;
  if (IS_MPI_ROOT_RANK) {
    options = parse_args(argc, argv);
    if (!options) {
      fprintf(stderr, "Argument parsing failed\n");
      exit(EXIT_FAILURE);
    }
  } else {
    options = malloc(sizeof(OPTIONS));
  }

  MPI_Bcast(options, sizeof(OPTIONS), MPI_BYTE, MPI_ROOT_RANK, MPI_COMM_WORLD);
  log_debug(
      "Rank %d: M: %ld, N: %ld, max_iteration: %ld, snapshot_frequency: %ld",
      world_rank, options->M, options->N, options->max_iteration,
      options->snapshot_frequency);

  M = options->M;
  N = options->N;
  max_iteration = options->max_iteration;
  snapshot_freq = options->snapshot_frequency;
  free(options);

  int dims[2] = {0, 0};
  MPI_Dims_create(world_size, 2, dims);
  m_processes = dims[0];
  n_processes = dims[1];
  /* if (M == N) { */
  /*   m_processes = n_processes = sqrt(world_size); */
  /* } else { */
  /*   if (M > N) { */
  /*     int mult = M / N; */
  /*     if (world_size % (mult + 1) != 0) { */
  /*       log_error("Number of processes must be divisible by %d", mult + 1);
   */
  /*       exit(EXIT_FAILURE); */
  /*     } */
  /*     n_processes = world_size / (mult + 1); */
  /*     m_processes = world_size / n_processes; */
  /*   } else { */
  /*     int mult = N / M; */
  /*     if (world_size % (mult + 1) != 0) { */
  /*       log_error("Number of processes must be divisible by %d", mult + 1);
   */
  /*       exit(EXIT_FAILURE); */
  /*     } */
  /*     m_processes = world_size / (mult + 1); */
  /*     n_processes = world_size / m_processes; */
  /*   } */
  /*   log_debug("M: %ld, N: %ld, m_processes: %d, n_processes: %d", M, N, */
  /*             m_processes, n_processes); */
  /* } */

  local_m = M / m_processes;
  local_n = N / n_processes;

  log_debug("Rank: %d: Local M: %d, Local N: %d", world_rank, local_m, local_n);

  MPI_Cart_create(MPI_COMM_WORLD, 2, (int[]){m_processes, n_processes},
                  (int[]){0, 0}, 0, &cart_comm);

  int cart_size;
  MPI_Comm_size(cart_comm, &cart_size);

  MPI_Comm_rank(cart_comm, &cart_rank);

  log_debug("Rank %d: Cart rank: %d", world_rank, cart_rank);

  MPI_Cart_coords(cart_comm, cart_rank, 2, local_cart_coords);

  log_debug("Rank %d: Cart rank: %d, Cart coords: (%d, %d)", world_rank,
            cart_rank, local_cart_coords[0], local_cart_coords[1]);

  local_m_offset = local_cart_coords[0] * (local_m + 2);
  local_n_offset = local_cart_coords[1] * (local_n + 2);

  // END: T3

  // Set up the initial state of the domain
  log_trace("Rank %d: Domain initialize", world_rank);
  domain_initialize();

  struct timeval t_start, t_end;

  // TASK: T2
  // Time your code
  // BEGIN: T2
  if (IS_MPI_ROOT_RANK)
    gettimeofday(&t_start, NULL);

  simulate();

  if (IS_MPI_ROOT_RANK) {
    gettimeofday(&t_end, NULL);
    printf("Total elapsed time: %lf seconds\n",
           WALLTIME(t_end) - WALLTIME(t_start));
  }
  // END: T2

  // Clean up and shut down
  domain_finalize();
  log_debug("Rank %d: Domain finalize", world_rank);

  // TASK: T1d
  // Finalise MPI
  // BEGIN: T1d
  MPI_Finalize();
  // END: T1d

  exit(EXIT_SUCCESS);
}
