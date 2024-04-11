import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def init_lattice(lattice_axis_length:int, ndim:int=2) -> np.ndarray:
    ''' Initialize a lattice of spins, with random orientation, of size N x N x ... x N (ndim times) where N is lattice_axis_length'''
    size = tuple([lattice_axis_length]*ndim)
    return np.random.choice([-1, 1], size=size)

def complete_hamiltonian(lattice:np.ndarray) -> float:
    ''' Calculate the Hamiltonian of the entire lattice '''
    coupling_constant = 1.0  
    external_field = 0.0
    hami = 0

    # must sum over nearest neighbors, with periodic boundary conditions
    # rolling the array (which loops back) aligns the neighbour along a direction with the central point, along each dimension 
    for dim in range(lattice.ndim):
        nearneighbours_sumterm = lattice * np.roll(lattice, 1, axis=dim) + lattice * np.roll(lattice, -1, axis=dim)
        hami += - coupling_constant * np.sum(nearneighbours_sumterm)
    hami = hami/2 # correcting for double counting of each pair of neighbours
    hami += - external_field * np.sum(lattice)

    return hami

def metropolis(lattice_old:np.ndarray, temperature:float, hamiltonian:float, ndim:int=2) -> np.ndarray:
    ''' Perform one step of the Metropolis algorithm.
        Flips a random spin, and accepts or rejects the flip based on the change in energy.
        Returns the updated lattice.
    '''
    lattice_new = lattice_old.copy()
    beta = 1.0 / temperature
    N = lattice_old.shape[0]

    slice_idx = [np.random.randint(0, N) for _ in range(ndim)]
    lattice_new[tuple(slice_idx)] *= -1
    lattice_axis_length = lattice_old.shape[0]
    # calculate the change in energy if we flip the spin at that point
    sum_neighbours = lattice_old[(slice_idx[0]+1) % lattice_axis_length, slice_idx[1]] \
                    + lattice_old[(slice_idx[0]-1) % lattice_axis_length, slice_idx[1]] \
                    + lattice_old[slice_idx[0], (slice_idx[1]+1) % lattice_axis_length] \
                    + lattice_old[slice_idx[0], (slice_idx[1]-1) % lattice_axis_length]
    delta_hami = - sum_neighbours * (lattice_new[tuple(slice_idx)] - lattice_old[tuple(slice_idx)])
    
    # reject the flip only if E increases with probability 1-exp(-beta * delta_E)
    if ((delta_hami > 0) & (np.exp(-beta * delta_hami) < np.random.rand())):
        return lattice_old, hamiltonian # reject flip if random number between 0 and 1 is greater than acceptance probability
    
    hamiltonian += delta_hami
    
    return lattice_new, hamiltonian

def find_equilibrium_energy(temperature:float) -> float:
    ''' Calculate the equilibrium energy of the lattice by running the Metropolis algorithm for a small 5x5 lattice.
        This equilibrium energy can be used to determine if the system is in equilibrium for larger systems. 
        Returns the average energy per particle that corresponds to equilibrium for a given temperature. 
    '''
    nsteps = 500 ; lattice_axis_length = 5 # small lattice equilibrates fast
    equilibrium_energies = []
    for i in range(5):
        hamiltonians = np.zeros(nsteps) # store energy at each step
        lattice = init_lattice(lattice_axis_length=lattice_axis_length, ndim=2) 
        hamiltonian = complete_hamiltonian(lattice)
        for i in range(nsteps):
            lattice, hamiltonian = metropolis(lattice, temperature, hamiltonian)
            hamiltonians[i] = hamiltonian
        equilibrium_energies.append(np.mean(hamiltonians[-nsteps//2:]) / lattice.size) # the average energy per particle of the last half of the steps is roughly the equilibrium energy
    return np.median(equilibrium_energies)

def equilibrate_lattice(lattice:np.ndarray, temperature:float, nsteps:int) -> np.ndarray:
    ''' Equilibrate the lattice by running the Metropolis algorithm until the energy is close to the equilibrium energy, or for nsteps steps.
        We define close as within +- 1% of the equilibrium energy. If this is not reached, we return the lattice after nsteps steps.
        Returns the lattice close to equilibrium, or after nsteps so that we can start the actual simulation from there.
    '''
    equilibrium_energy = find_equilibrium_energy(temperature=temperature)
    magnetizations = np.zeros(nsteps)
    hamiltonian = complete_hamiltonian(lattice)

    hamiltonians = np.zeros(nsteps)
    for i in range(nsteps): 
        lattice, hamiltonian = metropolis(lattice, temperature, hamiltonian)
        total_magnetization = np.sum(lattice)
        magnetizations[i] = total_magnetization
        hamiltonians[i] = hamiltonian

        if hamiltonian/lattice.size > equilibrium_energy - 0.01*np.abs(equilibrium_energy) \
                and hamiltonian/lattice.size < equilibrium_energy + 0.01*np.abs(equilibrium_energy):
            return lattice # stop the simulation if we are at equilibrium

    return lattice # if we don't reach equilibrium, return the lattice after nsteps

def autocorrelation_function(magnetizations):
    ''' Compute the autocorrelation_function for magnetizations (per spin), where time and time_max are taken from the length
        of magnetizations. Thus, give the array of magnetizations per spin from the equilibration time till time_max. '''
    time_max = len(magnetizations) - 1
    time = np.arange(len(magnetizations))
    chi = np.zeros_like(time)
    for t in time:
        chi[t] = 1/(time_max - t) * (np.sum( (magnetizations * np.roll(magnetizations, -t))[:time_max-t+1] ) \
                - np.sum(magnetizations[:time_max-t+1]) * np.sum(np.roll(magnetizations, -t)[:time_max-t+1]))
    return chi

def correlation_time(lattice:np.ndarray, temperature:float, nsteps_equi:int):
    step_factor = 50
    steps_per_round = nsteps_equi / step_factor

    magnetizations = np.zeros(steps_per_round)
    total_magnetizations = np.array([])
    hamiltonian = complete_hamiltonian(lattice)

    for i in range(nsteps_equi):
        lattice, hamiltonian = metropolis(lattice, temperature, hamiltonian)
        magnetizations[(i-1) % steps_per_round] = np.sum(lattice)

        if ((i-1) % steps_per_round == 0):
            total_magnetizations = np.append(total_magnetizations, magnetizations)
            autocor = autocorrelation_function(total_magnetizations / lattice.shape[0]**2)
            mask = (autocor <= 0)
            if len(autocor[mask]) == 0:
                pass
            else:
                idx_neg_autocor = np.argmax(mask)[0]
                cor_time = np.sum(autocor[:idx_neg_autocor] / autocor[0])
                return cor_time, lattice, hamiltonian, total_magnetizations
            
    return

def run_sim(lattice_axis_length:int, temperature:float, nsteps_equi:int, nsteps_sim:int) -> np.ndarray:
    ''' Run the simulation for a lattice of size lattice_axis_length x lattice_axis_length at temperature for nsteps steps, equilibrating for a maximum of nsteps_equi steps first.
        Returns the magnetization of the lattice at each step.
    '''
    lattice = init_lattice(lattice_axis_length=lattice_axis_length, ndim=2)
    lattice_equilibrated = equilibrate_lattice(lattice, temperature, nsteps_equi)

    cor_time, lattice, hamiltonian, magnetizations = correlation_time(lattice_equilibrated, temperature, nsteps_equi)
    ### %%%%%%%%%%% use these values to continue sim a certain number of cor_times and compute all averages and stds


    # now run the actual simulation to calculate properties near equilibrium
    magnetizations = np.zeros(nsteps_sim)
    hamiltonian = complete_hamiltonian(lattice_equilibrated)
    for i in range(nsteps_sim):
        lattice_equilibrated, hamiltonian = metropolis(lattice_equilibrated, temperature, hamiltonian)
        magnetizations[i] = np.sum(lattice_equilibrated)
    
    return magnetizations, cor_time

def main():
    ndim = 2 ; lattice_axis_length = 25 ; temperature = 1.5
    n_cycles = 300 # number of times we do N**2 Metropolis steps 
    num_spins = lattice_axis_length**ndim

    nsteps_sim = n_cycles * num_spins
    nsteps_equi = n_cycles * num_spins * 10 # maximum number of steps to equilibrate (if not reached earlier)

    # run the simulation a few times
    fig, ax = plt.subplots()
    for i in range(5):
        lattice = init_lattice(lattice_axis_length, ndim=2)
        magnetizations, cor_time = run_sim(lattice_axis_length, temperature, nsteps_equi, nsteps_sim)
        ax.plot(np.arange(n_cycles * num_spins) / num_spins, magnetizations / num_spins, label='Magnetization')
    ax.set_xlabel('Time [1/N$^2$]')
    ax.set_ylabel('Magnetization per spin')
    plt.show()

if __name__ == '__main__':
    main()