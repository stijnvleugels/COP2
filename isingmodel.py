import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def init_lattice(lattice_axis_length:int, ndim:int=2) -> np.ndarray:
    ''' Initialize a lattice of spins, with random orientation, of size N x N x ... x N (ndim times) where N is lattice_axis_length'''
    size = tuple([lattice_axis_length]*ndim)
    return np.random.choice([-1, 1], size=size)

def hamiltonian(lattice:np.ndarray) -> float:
    ''' Calculate the Hamiltonian of the lattice '''
    coupling_constant = 1.0  
    external_field = 0.0
    hami = 0

    # must sum over nearest neighbors, with periodic boundary conditions
    # rolling the array (which loops back) aligns the neighbour along a direction with the central point, along each dimension 
    for dim in range(lattice.ndim):
        nearneighbours_sumterm = lattice * np.roll(lattice, 1, axis=dim) + lattice * np.roll(lattice, -1, axis=dim)
        hami += - coupling_constant * np.sum(nearneighbours_sumterm)
    hami +=  - external_field * np.sum(lattice)

    return hami

def metropolis(lattice_old:np.ndarray, temperature:float, ndim:int=2) -> np.ndarray:
    ''' Perform one step of the Metropolis algorithm.
        Flips a random spin, and accepts or rejects the flip based on the change in energy.
        Returns the updated lattice.
    '''
    lattice_new = lattice_old.copy()
    beta = 1.0 / temperature
    N = lattice_old.shape[0]

    slice_idx = [np.random.randint(0, N) for _ in range(ndim)]
    # calculate the change in energy if we flip the spin at that point
    hami_old = hamiltonian(lattice_old)
    lattice_new[tuple(slice_idx)] *= -1
    hami_new = hamiltonian(lattice_new)
    
    # accept the flip if E decreases, or with probability exp(-beta * delta_E)
    delta_hami = hami_new - hami_old
    if delta_hami <= 0:
        return lattice_new # accept flip
    else:
        if np.exp(-beta * delta_hami) < np.random.rand():
            return lattice_old # reject flip if random number is greater than acceptance probability
        
    return lattice_new

def equilibrate(lattice:np.ndarray, temperature:float, nsteps:int) -> np.ndarray:
    ''' Equilibrate the lattice by running the Metropolis algorithm for nsteps steps.
        Returns the magnetization of the lattice at each step.
    '''
    magnetizations = np.zeros(nsteps)
    for i in tqdm(range(nsteps)):
        lattice = metropolis(lattice, temperature)
        total_magnetization = np.sum(lattice)
        magnetizations[i] = total_magnetization

    return magnetizations

def main():
    ndim = 2
    lattice_axis_length = 20
    temperature = 5
    n_cycles = 100 # number of times we do N**2 Metropolis steps 

    num_spins = lattice_axis_length**ndim

    fig, ax = plt.subplots()
    for i in range(3): # do 3 runs because it can get stuck in a local minimum
        lattice = init_lattice(lattice_axis_length, ndim=2)
        magnetizations = equilibrate(lattice, temperature, n_cycles * num_spins)
        ax.plot(np.arange(n_cycles * num_spins) / num_spins, magnetizations / num_spins, label='Magnetization')
    ax.set_xlabel('Time [1/N^2]')
    ax.set_ylabel('Magnetization per spin')
    plt.show()

    # imshow the last lattice
    fig, ax = plt.subplots()
    ax.imshow(lattice, cmap='gray', vmin=-1, vmax=1)
    plt.show()

    # TODO: 
    # - run for different temperatures (1 to 4 with 0.2 increments)
    # - sanity check if the outcomes are ok if we plot it for different temperatures
    # - find a way to check if the system is in equilibrium, so we can stop the simulation instead of running for a fixed number of steps
    # - plot the magnetization as a function of temperature
    # - ...


if __name__ == '__main__':
    main()