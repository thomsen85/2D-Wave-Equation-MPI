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

// Types
MPI_Datatype column_type;
MPI_Datatype file_view;
MPI_Datatype file_grid;

#define MPI_ROOT_RANK 0

// Checks
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

  for (int_t i = local_m_offset; i < local_m_offset + local_m; i++) {
    for (int_t j = local_n_offset; j < local_n_offset + local_n; j++) {
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
    // Send Left Column

    log_trace("Rank: %d (%d, %d): Sending left border to %d...", world_rank,
              local_cart_coords[0], local_cart_coords[1], left_rank);
    MPI_Send(&U(0, 0), 1, column_type, left_rank, 0, cart_comm);
    log_trace("Rank: %d: Sendt left border", world_rank);
  }

  // Not Rightmost
  if (!IS_MPI_RIGHTMOST) {
    // Recive left borders and insert into right borders
    log_trace("Rank: %d: Waiting to Recv left border from right neighbour",
              world_rank);
    MPI_Recv(&U(0, local_n), 1, column_type, right_rank, 0, cart_comm,
             MPI_STATUS_IGNORE);
    log_trace("Rank: %d: Recv left border inserting into right border...",
              world_rank);

    log_trace("Rank: %d: Sending Right Border to right neighbour %d...",
              world_rank, right_rank);
    // Send Right Column
    MPI_Send(&U(0, local_n - 1), 1, column_type, right_rank, 0, cart_comm);
    /* free(send_column); */
    log_trace("Rank: %d: Done sending", world_rank);
  }

  // Recive left borders
  // Not Leftmost
  if (!IS_MPI_LEFTMOST) {
    /* real_t *recv_column = malloc(local_m * sizeof(real_t)); */

    log_trace("Rank: %d: Recieving Right border from left neighbour...",
              world_rank);
    MPI_Recv(&U(0, -1), 1, column_type, left_rank, 0, cart_comm,
             MPI_STATUS_IGNORE);
    log_trace("Rank: %d: Recieved Right border from left neighbour",
              world_rank);
  }

  // Send top borders
  // Not Topmost
  if (!IS_MPI_TOPMOST) {
    // Send Top Row
    MPI_Send(&U(0, 0), local_n, MPI_DOUBLE, top_rank, 0, cart_comm);
    log_trace("Rank: %d: Sendt top border", world_rank);
  }

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
    log_trace("Rank: %d: Recv top border", world_rank);
  }
  int source, above, under, left, right;

  // Get the above neighbour
  MPI_Cart_shift(comm_cart, 0, -1, &source, &above);

  // Get the below neighbour
  MPI_Cart_shift(comm_cart, 0, 1, &source, &under);

  // Get the left neighbour
  MPI_Cart_shift(comm_cart, 1, -1, &source, &left);

  // Get the right neighbour
  MPI_Cart_shift(comm_cart, 1, 1, &source, &right);

  // Exchange columns (left and right)
  MPI_Sendrecv(&U(0, 0), 1, column_type, left, 0, &U(0, local_N), 1,
               column_type, right, 0, comm_cart, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&U(0, local_N - 1), 1, column_type, right, 0, &U(0, -1), 1,
               column_type, left, 0, comm_cart, MPI_STATUS_IGNORE);
  // Exchange rows (above and under)
  MPI_Sendrecv(&U(0, 0), local_N, MPI_DOUBLE, under, 0, &U(local_M, 0), local_M,
               MPI_DOUBLE, above, 0, comm_cart, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&U(local_M - 1, 0), local_N, MPI_DOUBLE, above, 0, &U(-1, 0),
               local_N, MPI_DOUBLE, under, 0, comm_cart, MPI_STATUS_IGNORE);
  // Send and receive data from the top and bottom
  // END: T6
}

// TASK: T7
// Neumann (reflective) boundary condition
void boundary_condition(void) {
  // BEGIN: T7
  // Columns
  for (int_t i = 0; i < local_m; i++) {
    if (IS_MPI_LEFTMOST)
      U(i, -1) = U(i, 1);

    if (IS_MPI_RIGHTMOST)
      U(i, local_n) = U(i, local_n - 2);
  }
  // Rows
  for (int_t j = 0; j < local_n; j++) {
    if (IS_MPI_TOPMOST)
      U(-1, j) = U(1, j);

    if (IS_MPI_BOTTOMMOST)
      U(local_m, j) = U(local_m - 2, j);
  }
  // END: T7
}

void setup_types() {
  // Column type
  MPI_Type_vector(local_m, 1, local_n + 2, MPI_DOUBLE, &column_type);
  MPI_Type_commit(&column_type);

  // File Grid Type
  int full_array_size[2] = {local_m + 2, local_n + 2};
  int inner_array_size[2] = {local_m, local_n};
  int offset[2] = {0, 0};

  MPI_Type_create_subarray(2, full_array_size, inner_array_size, offset,
                           MPI_ORDER_C, MPI_DOUBLE, &file_grid);
  MPI_Type_commit(&file_grid);

  // File View Type
  int file_size[2] = {M, N};
  int file_grid_size[2] = {local_m, local_n};
  int view_offset[2] = {local_m * local_cart_coords[0],
                        local_n * local_cart_coords[1]};

  MPI_Type_create_subarray(2, file_size, file_grid_size, view_offset,
                           MPI_ORDER_C, MPI_DOUBLE, &file_view);
  MPI_Type_commit(&file_view);
}

// TASK: T8
// Save the present time step in a numbered file under 'data/'
void domain_save(int_t step) {
  // BEGIN: T8
  // Inspiration is taken from
  // https://stackoverflow.com/questions/33537451/writing-distributed-arrays-using-mpi-io-and-cartesian-topology

  /* MPI IO */
  MPI_File fh;

  char filename[256];
  sprintf(filename, "data/%.5ld.dat", step);
  log_info("Rank %d: Writing to file: %s", world_rank, filename);

  MPI_File_open(cart_comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fh);

  MPI_File_set_view(fh, 0, MPI_DOUBLE, file_view, "native", MPI_INFO_NULL);
  MPI_File_write_all(fh, &buffers[1][0], 1, file_grid, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);
  // END: T8
}

// Main time integration.
void simulate(void) {
  // Go through each time step
  for (int_t iteration = 0; iteration <= max_iteration; iteration++) {
    log_trace("Rank %d: DONE --- Iteration: %ld", world_rank, iteration);
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
  log_set_level(LOG_ERROR);
  // TASK: T1c
  // Initialise MPI
  // BEGIN: T1c
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // END: T1c

  // TODO: Remove this
  char log_file[256];
  sprintf(log_file, "log_rank_%d.log", world_rank);
  FILE *log_fp = fopen(log_file, "w");
  if (!log_fp) {
    fprintf(stderr, "Failed to open log file\n");
    exit(EXIT_FAILURE);
  }
  log_add_fp(log_fp, LOG_TRACE);

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

  int dims[2] = {0, 0};
  MPI_Dims_create(world_size, 2, dims);
  m_processes = dims[0];
  n_processes = dims[1];

  local_m = M / m_processes;
  local_n = N / n_processes;

  log_debug("Rank: %d: M: %ld, N: %ld, m_processes: %d, n_processes: %d",
            world_rank, M, N, m_processes, n_processes);
  log_debug("Rank: %d: Local_m: %d, Local_n: %d", world_rank, local_m, local_n);

  MPI_Cart_create(MPI_COMM_WORLD, 2, (int[]){m_processes, n_processes},
                  (int[]){0, 0}, 0, &cart_comm);

  int cart_size;
  MPI_Comm_size(cart_comm, &cart_size);
  MPI_Comm_rank(cart_comm, &cart_rank);

  MPI_Cart_coords(cart_comm, cart_rank, 2, local_cart_coords);

  log_debug("Rank %d: Cart rank: %d, Cart coords: (%d, %d), Cart size",
            world_rank, cart_rank, local_cart_coords[0], local_cart_coords[1],
            cart_size);

  local_m_offset = local_cart_coords[0] * local_m;
  local_n_offset = local_cart_coords[1] * local_n;
  log_debug("Rank %d: Local_m_offset: %d, Local_n_offset: %d", world_rank,
            local_m_offset, local_n_offset);

  // END: T3

  // Set up the initial state of the domain
  log_trace("Rank %d: Domain initializing", world_rank);
  domain_initialize();

  struct timeval t_start, t_end;
  setup_types();

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
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Type_free(&file_view);
  MPI_Type_free(&file_grid);
  MPI_Type_free(&column_type);

  MPI_Comm_free(&cart_comm);
  MPI_Finalize();
  // END: T1d

  free(options);
  // TODO: Remove this
  fclose(log_fp);

  exit(EXIT_SUCCESS);
}
