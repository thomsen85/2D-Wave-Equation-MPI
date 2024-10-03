#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

#include "argument_utils.h"


// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Option to change numerical precision
typedef int64_t int_t;
typedef double real_t;

// Simulation parameters: size, step count, and how often to save the state
int_t
    N = 256,
    M = 256,
    max_iteration = 4000,
    snapshot_freq = 20;

// Wave equation parameters, time step is derived from the space step
const real_t
    c  = 1.0,
    dx = 1.0,
    dy = 1.0;
real_t
    dt;

// Buffers for three time steps, indexed with 2 ghost points for the boundary
real_t
    *buffers[3] = { NULL, NULL, NULL };

#define U_prv(i,j) buffers[0][((i)+1)*(N+2)+(j)+1]
#define U(i,j)     buffers[1][((i)+1)*(N+2)+(j)+1]
#define U_nxt(i,j) buffers[2][((i)+1)*(N+2)+(j)+1]


// Rotate the time step buffers.
void move_buffer_window ( void )
{
    real_t *temp = buffers[0];
    buffers[0] = buffers[1];
    buffers[1] = buffers[2];
    buffers[2] = temp;
}


// Save the present time step in a numbered file under 'data/'
void domain_save ( int_t step )
{
    char filename[256];
    sprintf ( filename, "data/%.5ld.dat", step );
    FILE *out = fopen ( filename, "wb" );
    for ( int_t i=0; i<M; i++ )
    {
        fwrite ( &U(i,0), sizeof(real_t), N, out );
    }
    fclose ( out );
}


// Set up our three buffers, and fill two with an initial perturbation
void
domain_initialize ( void )
{
    buffers[0] = malloc ( (M+2)*(N+2)*sizeof(real_t) );
    buffers[1] = malloc ( (M+2)*(N+2)*sizeof(real_t) );
    buffers[2] = malloc ( (M+2)*(N+2)*sizeof(real_t) );

    for ( int_t i=0; i<M; i++ )
    {
        for ( int_t j=0; j<N; j++ )
        {
            // Calculate delta (radial distance) adjusted for M x N grid
            real_t delta = sqrt ( ((i - M/2.0) * (i - M/2.0)) / (real_t)M +
                                ((j - N/2.0) * (j - N/2.0)) / (real_t)N );
            U_prv(i,j) = U(i,j) = exp ( -4.0*delta*delta );
        }
    }

    // Set the time step for 2D case
    dt = dx*dy / (c * sqrt (dx*dx+dy*dy));
}


// Get rid of all the memory allocations
void domain_finalize ( void )
{
    free ( buffers[0] );
    free ( buffers[1] );
    free ( buffers[2] );
}


// Integration formula (Eq. 9 from the pdf document)
void time_step ( void )
{
    for ( int_t i=0; i<M; i++ )
    {
        for ( int_t j=0; j<N; j++ )
        {
            U_nxt(i,j) = -U_prv(i,j) + 2.0*U(i,j)
                     + (dt*dt*c*c)/(dx*dy) * (
                        U(i-1,j)+U(i+1,j)+U(i,j-1)+U(i,j+1)-4.0*U(i,j)
                     );
        }
    }
}


// Neumann (reflective) boundary condition
void boundary_condition ( void )
{
    for ( int_t i=0; i<M; i++ )
    {
        U(i,-1) = U(i,1);
        U(i,N)  = U(i,N-2);
    }
    for ( int_t j=0; j<N; j++ )
    {
        U(-1,j) = U(1,j);
        U(M,j)  = U(M-2,j);
    }
}


// Main time integration.
void simulate( void )
{
    // Go through each time step
    for ( int_t iteration=0; iteration<=max_iteration; iteration++ )
    {
        if ( (iteration % snapshot_freq)==0 )
        {
            domain_save ( iteration / snapshot_freq );
        }

        // Derive step t+1 from steps t and t-1
        boundary_condition();
        time_step();

        // Rotate the time step buffers
        move_buffer_window();
    }
}


int main ( int argc, char **argv )
{
    OPTIONS *options = parse_args( argc, argv );
    if ( !options )
    {
        fprintf( stderr, "Argument parsing failed\n" );
        exit( EXIT_FAILURE );
    }

    M = options->M;
    N = options->N;
    max_iteration = options->max_iteration;
    snapshot_freq = options->snapshot_frequency;

    // Set up the initial state of the domain
    domain_initialize();

    struct timeval t_start, t_end;

    gettimeofday ( &t_start, NULL );
    simulate();
    gettimeofday ( &t_end, NULL );

    printf ( "Total elapsed time: %lf seconds\n",
        WALLTIME(t_end) - WALLTIME(t_start)
    );

    // Clean up and shut down
    domain_finalize();
    exit ( EXIT_SUCCESS );
}
