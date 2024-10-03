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

#define MPI_ROOT 0
#define IS_MPI_ROOT (world_rank == MPI_ROOT)
#define IS_MPI_LAST (world_rank == world_size - 1)

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
  buffers[0] = malloc((M + 2) * (N + 2) * sizeof(real_t));
  buffers[1] = malloc((M + 2) * (N + 2) * sizeof(real_t));
  buffers[2] = malloc((M + 2) * (N + 2) * sizeof(real_t));

  for (int_t i = 0; i < M; i++) {
    for (int_t j = 0; j < N; j++) {
      // Calculate delta (radial distance) adjusted for M x N grid
      real_t delta = sqrt(((i - M / 2.0) * (i - M / 2.0)) / (real_t)M +
                          ((j - N / 2.0) * (j - N / 2.0)) / (real_t)N);
      U_prv(i, j) = U(i, j) = exp(-4.0 * delta * delta);
    }
  }

  // Set the time step for 2D case
  dt = dx * dy / (c * sqrt(dx * dx + dy * dy));
  // END: T4
}

// Get rid of all the memory allocations
void domain_finalize(void) {
  free(buffers[0]);
  free(buffers[1]);
  free(buffers[2]);
}

// TASK: T5
// Integration formula
void time_step(void) {
  // BEGIN: T5
  for (int_t i = 0; i < M; i++) {
    for (int_t j = 0; j < N; j++) {
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
  ;
  // END: T6
}

// TASK: T7
// Neumann (reflective) boundary condition
void boundary_condition(void) {
  // BEGIN: T7
  for (int_t i = 0; i < M; i++) {
    U(i, -1) = U(i, 1);
    U(i, N) = U(i, N - 2);
  }
  for (int_t j = 0; j < N; j++) {
    U(-1, j) = U(1, j);
    U(M, j) = U(M - 2, j);
  }
  // END: T7
}

// TASK: T8
// Save the present time step in a numbered file under 'data/'
void domain_save(int_t step) {
  // BEGIN: T8
  char filename[256];
  sprintf(filename, "data/%.5ld.dat", step);
  FILE *out = fopen(filename, "wb");
  for (int_t i = 0; i < M; i++) {
    fwrite(&U(i, 0), sizeof(real_t), N, out);
  }
  fclose(out);
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
  if (IS_MPI_ROOT) {
    options = parse_args(argc, argv);
    if (!options) {
      fprintf(stderr, "Argument parsing failed\n");
      exit(EXIT_FAILURE);
    }
  } else {
    options = malloc(sizeof(OPTIONS));
  }

  MPI_Bcast(options, sizeof(OPTIONS), MPI_BYTE, MPI_ROOT, MPI_COMM_WORLD);
  log_debug(
      "Rank: %d, M: %ld, N: %ld, max_iteration: %ld, snapshot_frequency: %ld",
      world_rank, options->M, options->N, options->max_iteration,
      options->snapshot_frequency);

  M = options->M;
  N = options->N;
  max_iteration = options->max_iteration;
  snapshot_freq = options->snapshot_frequency;
  // END: T3

  // Set up the initial state of the domain
  domain_initialize();

  struct timeval t_start, t_end;

  // TASK: T2
  // Time your code
  // BEGIN: T2
  if (IS_MPI_ROOT)
    gettimeofday(&t_start, NULL);

  simulate();

  if (IS_MPI_ROOT) {
    gettimeofday(&t_end, NULL);
    printf("Total elapsed time: %lf seconds\n",
           WALLTIME(t_end) - WALLTIME(t_start));
  }
  // END: T2

  // Clean up and shut down
  domain_finalize();

  // TASK: T1d
  // Finalise MPI
  // BEGIN: T1d
  MPI_Finalize();
  // END: T1d

  exit(EXIT_SUCCESS);
}
