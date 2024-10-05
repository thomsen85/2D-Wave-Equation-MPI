1. MPI_Gather collects data from all processes and sends it to a single root process, which could become a bottleneck if say we were going to write to a file or send somewhere else.
   The root process has to handle the received data and could end up with having to handle I/O on its own.
   MPI Allgather collects data from all processes and distributes it to all processes, this involves much more communication because every process needs to receive all the data, so more data needs to be sendt around.
   This can impact performance in programs with many processes. In conclution i would say that it depends on the use case of the program but generally MPI Gather is more efficient with the data,
   while MPI Allgather can be useful when all processes need the complete dataset.

2. To communicate the diagonal neighbour, you realy dont need to do anything special. Lets take an example with 2x2 grid and 1 padding.

´´´
x x x  x x x
x 1 x  x 2 x
x x x  x x x

x x x  x x x
x 3 x  x 4 x    
x x x  x x x
´´´
First lets communicate the vertical borders. Here we only send that data outside the ghost cells.

x x x  x x x
x 1 x  x 2 x
x 3 x  x 4 x

x 1 x  x 2 x
x 3 x  x 4 x
x x x  x x x

Now communuicate the horizontal borders. But also send the ghost cells.

So we get:

x x x  x x x
x 1 2  1 2 x
x 3 4  3 4 x

x 1 2  1 2 x
x 3 4  3 4 x
x x x  x x x


3. Running the code with M = 2048 and N = 512. 
TODO:
4. Weak scaling refers to increasing the problem size proportionally with the number of processors. 
  The goal is to maintain a constant workload per processor, so performance is measured by how well the system handles larger problems as more processors are added. 
  In strong scaling, the problem size remains constant while more processors are added. The goal here is to reduce the execution time, 
  and the focus is on how well the system distributes the fixed amount of work across more processors.
  Source: https://hpc-wiki.info/hpc/Scaling#Strong_Scaling
