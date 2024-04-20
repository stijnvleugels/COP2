This set of python files allows you to run a 2D ising model with optional external magnetic field and plot the results. 

Run the simulation in the terminal with the following command: \
`python isingmodel.py [N] [filename] [H] [seed]` \
it is possible to skip the arguments at the end, they will be set to defaults, but it's impossible to skip an argument and then set the next one in this order.

Producing the plots is done using: \
`python plots.py [filename]`

The arguments have the following meaning: 
 - N is the lattice axis length. E.g., N=50 yields a 50x50 lattice of spins.
 - filename is the name of the file the results will be written to, excluding extension (it will write to .csv)
 - H is the strength of the external magnetic field, in the units 1/J 
 - seed is the random seed (integer). Results in the report were produced with seed 67.

 The file will automatically run for all T in range [1,4] with steps of 0.2, with 3 simulations at each temperature. Note that near the critical temperature, the simulation will slow down significantly, seriously increasing runtime. 



