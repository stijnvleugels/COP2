import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import matplotlib as mpl

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
    lattice_axis_length = lattice_old.shape[0]
    
    slice_idx = [np.random.randint(0, lattice_axis_length) for _ in range(ndim)]
    lattice_new[tuple(slice_idx)] *= -1
    
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
    nsteps = 1000 ; lattice_axis_length = 5 # small lattice equilibrates fast
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
            return lattice, magnetizations[:i+1], hamiltonians[:i+1] # stop the simulation if we are at equilibrium

    return lattice, magnetizations, hamiltonians # if we don't reach equilibrium, return the lattice after nsteps

def autocorrelation_function(magnetizations:np.ndarray, sampling_factor:int=500, t_ratio_threshold:float=0.05) -> tuple[np.ndarray, float]:
    ''' Compute the autocorrelation chi for magnetizations.
        Calculates chi `sampling_factor` times over the timespan `len(magnetisations) * t_ratio_threshold` to reduce computation time, and interpolates linearly between these points to get chi(t) at all timesteps.
        Returns chi(t) and t[chi<0]/t_max, the fraction of the total time at which chi becomes negative.
    '''
    time_fraction = t_ratio_threshold * 1.3 # if the point chi<0 has not been found before the threshold, don't bother calculating further (t/t_max will be too low for sufficient samples)
    time_max = len(magnetizations) - 1
    time = np.arange(len(magnetizations)*time_fraction, dtype=int)
    spacing = len(time) // sampling_factor
    chi = np.zeros( int(len(time) // spacing * 1.1 ), dtype=float) # 1.1 is a buffer for rounding with spacing slicing

    for t in time[:-1:spacing]:
        sliced_magnetizations = magnetizations[:time_max-t]
        rolled_magnetizations = np.roll(magnetizations, -t)[:time_max-t]
        cross_term = 1/(time_max - t) * np.sum(sliced_magnetizations * rolled_magnetizations)
        meansquared_term = 1/(time_max -t) * np.sum(sliced_magnetizations) * 1/(time_max - t) * np.sum(rolled_magnetizations)
        chi[t//spacing] = cross_term - meansquared_term
        if chi[t//spacing] < 0:
            # print('True found negative chi at t/time_max = ', t/time_max)
            # interpolate linearly to get the autocorrelation function at all timesteps (between spacings)
            chi_new = np.interp(time[:t], time[:-1:spacing][:t//spacing], chi[:t//spacing])
            # fig, ax = plt.subplots()
            # ax.plot(time[:t] / 50**2, chi_new / chi_new[0], linestyle='-', label='Interpolated $\chi(t)$', zorder=1, color='blue')
            # ax.scatter(time[:-1:spacing][:t//spacing] / 50**2, chi[:t//spacing] / chi_new[0], label='$\chi$ samples', s=5, marker='x', color='black', zorder=3)
            # tcor = np.sum(chi_new/chi_new[0])
            # ax.plot(np.arange(len(chi_new))/50**2, np.exp(-np.arange(len(chi_new))/tcor), linestyle='--',label=r'exp(-t/$\tau$)', zorder=2, color='green')
            # ax.set_xlabel('t [1/N$^2$]', fontsize=16)
            # ax.set_ylabel('$\chi(t) / \chi(0)$', fontsize=16)
            # # set tick label size to 12
            # ax.tick_params(axis='both', which='major', labelsize=12)
            # legend = ax.legend(loc='upper right', fontsize='x-large', markerscale=4)
            # plt.tight_layout()
            # ax.grid(alpha=0.5)
            # plt.savefig('autocorrelation_vs_t22.pdf')
            # plt.show()
            return chi_new, t/time_max
    
    chi_new = np.interp(time[:t], time[:-1:spacing][:t//spacing], chi[:t//spacing])
    return chi_new, 1

def estimate_chi_zero(magnetizations:np.ndarray, sampling_factor:int=10, t_ratio_threshold:float=0.05) -> float:
    ''' Estimate the time at which the autocorrelation function becomes negative with very crude sampling.
        This helps narrow down the range of time for which to calculate the autocorrelation.
    '''
    _, t_ratio = autocorrelation_function(magnetizations, sampling_factor, t_ratio_threshold)
    return t_ratio

def run_sim(lattice:np.ndarray, temperature:float, num_boxes:int):
    t_ratio_threshold = 0.05
    step_factor = 100 # check for correlation time every 1% of the maximum number of steps

    max_steps_sim = 1000000 * 16 * num_boxes # maximum number of steps - runs with this if no correlation time is found
    steps_per_round = max_steps_sim // step_factor

    magnetizations = np.zeros(max_steps_sim)
    hamiltonian = complete_hamiltonian(lattice)
    hamiltonians = np.zeros(max_steps_sim)
    for i in range(max_steps_sim):
        lattice, hamiltonian = metropolis(lattice, temperature, hamiltonian)
        magnetizations[i] = np.sum(lattice) 
        hamiltonians[i] = hamiltonian
        if ((i+1) % steps_per_round == 0):
            round_num = (i+1) // steps_per_round
            t_ratio_chi_subzero = estimate_chi_zero(magnetizations[:i+1], 10*(round_num//10 + 1), t_ratio_threshold) # every 10 rounds, increase sampling factor because we have more total steps
            if t_ratio_chi_subzero < t_ratio_threshold: # if t_max is not much larger than the t at which chi becomes negative, there is lots of noise and no proper exponential shape
                autocor, _ = autocorrelation_function(magnetizations[:i+1] / lattice.size, sampling_factor=100, t_ratio_threshold=t_ratio_chi_subzero)
                cor_time = np.sum(autocor/autocor[0])
                # print(f'Correlation time: {cor_time}')
                # fig, ax = plt.subplots()
                # ax.plot(autocor/autocor[0], label='Autocorrelation function')
                # ax.plot(np.arange(len(autocor)), np.exp(-np.arange(len(autocor))/cor_time), linestyle='--',label='exp(-t/tau)')
                # ax.set(xlabel='t', ylabel='Autocorrelation function', title='Autocorrelation function vs t')
                # ax.legend()
                # plt.show()

                needed_nsteps = round(16 * num_boxes * cor_time - (i+1) + 1)
                if (needed_nsteps > 0):
                    print('Running remainder of simulation given correlation time...')
                    for j in range(needed_nsteps+1):
                        lattice, hamiltonian = metropolis(lattice, temperature, hamiltonian)
                        magnetizations[i+j+1] = np.sum(lattice) 
                        hamiltonians[i+j+1] = hamiltonian

                    return cor_time, lattice, hamiltonians[:i+j+2], magnetizations[:i+j+2]

                return cor_time, lattice, hamiltonians[:i+1], magnetizations[:i+1]
            
    return cor_time, lattice, hamiltonians, magnetizations

def init_sim(lattice_axis_length:int, temperature:float, nsteps_equi:int, num_boxes:int) -> np.ndarray:
    ''' Run the simulation for a lattice of size lattice_axis_length x lattice_axis_length at temperature for nsteps steps, equilibrating for a maximum of nsteps_equi steps first.
        Returns the magnetization of the lattice at each step.
    '''
    lattice = init_lattice(lattice_axis_length=lattice_axis_length, ndim=2)
    print('Equilibrating lattice...')
    lattice_equilibrated = equilibrate_lattice(lattice, temperature, nsteps_equi)[0]

    print('Calculating correlation time...')
    cor_time, lattice, hamiltonians, magnetizations = run_sim(lattice_equilibrated, temperature, num_boxes)

    return magnetizations, hamiltonians, cor_time

def standard_deviation(quantity:np.ndarray, cor_time:float):
    '''Takes a quantity measured for each lattice configuration and the correlation time of the simulation.
       Returns the standard deviation, taking into account that there is only a certain amount of independent measurements.'''
    time_max = len(quantity) - 1
    std = np.sqrt( 2*cor_time / time_max * ( np.mean(quantity**2) - np.mean(quantity)**2 ) )
    return std

def magnetic_susceptibility_block(magnetizations:np.ndarray, num_spins:float, temperature:float):
    '''Computing the magnetic susceptibility within a block. Given magnetizations are only those within the block.'''
    beta = 1.0 / temperature
    return beta / num_spins * ( np.mean(magnetizations**2) - np.mean(magnetizations)**2 )

def specific_heat_per_spin_block(hamiltonians:np.ndarray, num_spins:float, temperature:float):
    '''Computing the specific heat per spin within a block. Given hamiltonians are only those within the block.'''
    k_B = 1.0
    return ( np.mean(hamiltonians**2) - np.mean(hamiltonians)**2 ) / (num_spins * k_B * temperature**2)

def quantities(magnetizations:np.ndarray, hamiltonians:np.ndarray, num_spins:float, cor_time:float, temperature:float, filename):
    '''Computing the energy per spin, absolute spin, magnetic susceptibility and specific heat per spin after a
       simulation is run.'''

    absolute_spin = np.abs(magnetizations) / num_spins
    energy_per_spin = hamiltonians / num_spins

    total_time = len(magnetizations)
    block_time = 16*cor_time
    right_block_edges = np.arange(block_time, total_time, block_time)
    magnetic_susceptibility = np.zeros(len(right_block_edges))
    specific_heat_per_spin = np.zeros(len(right_block_edges))

    # walking over blocks each with length of 16 cor_times
    left_edge = 0
    for i, right_edge in enumerate(right_block_edges):
        right_edge = round(right_edge)
        magnetic_susceptibility[i] = magnetic_susceptibility_block(magnetizations[left_edge:right_edge], num_spins, temperature)
        specific_heat_per_spin[i] = specific_heat_per_spin_block(hamiltonians[left_edge:right_edge], num_spins, temperature)
        left_edge = right_edge

    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([temperature, cor_time,
                            np.mean(absolute_spin), standard_deviation(absolute_spin, cor_time),
                            np.mean(energy_per_spin), standard_deviation(energy_per_spin, cor_time),
                            np.mean(magnetic_susceptibility), np.std(magnetic_susceptibility),
                            np.mean(specific_heat_per_spin), np.std(specific_heat_per_spin)])
        csvfile.close()

    return

def plot_equi(nsteps_equi:int):
    colors = ['blue', 'black', 'red']
    alphas = [1, 0.7]

    # fig, ax = plt.subplots(2,1,figsize=(8,6), constrained_layout=True, sharex=True)
    # for j, temperature in enumerate([1.0, 2.2, 4.0]):
    #     for i in range(1):
    #         print('bezig')
    #         axis_length=50
    #         lattice = init_lattice(lattice_axis_length=axis_length, ndim=2)
    #         lattice_equilibrated, magnetizations, hamiltonians = equilibrate_lattice(lattice, temperature, nsteps_equi)
    #         ax[0].plot(np.arange(len(magnetizations)) / axis_length**2, magnetizations / axis_length**2, label=f'T={temperature}', color=colors[j], alpha=0.7)
    #         ax[1].plot(np.arange(len(hamiltonians)) / axis_length**2, hamiltonians / axis_length**2, color=colors[j], alpha=0.7)

    # ax[0].set(ylabel='magnetization per spin', xscale='log')
    # ax[1].set(xlabel='t [$1/N^2$]', ylabel='energy per spin', xscale='log')
    # ax[0].legend(loc='best')
    # plt.savefig('Equi.pdf')
    # plt.show()

    cmap = mpl.colors.ListedColormap(['black', 'white'])
    bounds=[-1,0,1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(1,3,figsize=(8,4), constrained_layout=True, sharey=True)
    for j, temperature in enumerate([1.0, 2.2, 4.0]):
        for i in range(1):
            print('bezig')
            axis_length=50
            lattice = init_lattice(lattice_axis_length=axis_length, ndim=2)
            lattice_equilibrated, magnetizations, hamiltonians = equilibrate_lattice(lattice, temperature, nsteps_equi)
            a=ax[j].imshow(lattice_equilibrated, cmap=cmap, norm=norm)
            ax[j].set(xlabel='x spins', title=f'T={temperature}')
    ax[0].set(ylabel='y spins')

    plt.colorbar(a, cmap=cmap, norm=norm, boundaries=bounds, fraction=0.046, pad=0.04)
    plt.savefig('lattices.pdf')
    plt.show()

    return

def main():
    np.random.seed(67)
    ndim = 2 ; lattice_axis_length = 50
    n_cycles = 300 # number of times we do N**2 Metropolis steps 
    num_spins = lattice_axis_length**ndim
    
    num_boxes = 10
    nsteps_equi = n_cycles * num_spins * 10 # maximum number of steps to equilibrate (if not reached earlier)

    # plot_equi(nsteps_equi)


    # # initialise csv file to save the simulation
    # filename = "simulation.csv"
    # field_names = ['Temperature', 'Correlation_time', 
    #                'Magnetization_mean', 'Magnetization_std', 
    #                'Energy_mean', 'Energy_std', 
    #                'Magnetic_susceptibility_mean', 'Magnetic_susceptibility_std',
    #                'Specific_heat_mean', 'Specific_heat_std']
    # with open(filename, 'w') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(field_names)
    #     csvfile.close()

    # # for temperature in np.arange(1.0, 4.1, 0.2):
    # fig, ax = plt.subplots(1,1)
    # for temperature in [1.0, 2.0, 3.0, 4.0]:
    #     print('Running simulation for T = ', temperature)
    #     for i in range(1):
    #         magnetizations, hamiltonians, cor_time = init_sim(lattice_axis_length, temperature, nsteps_equi, num_boxes)
    #         # quantities(magnetizations, hamiltonians, num_spins, cor_time, temperature, filename)
    #         ax.plot(np.arange(len(magnetizations)) / num_spins, magnetizations / num_spins, label=f'T={temperature}, N={lattice_axis_length}')
    # ax.set(xlabel='t [$1/N^2$]', ylabel='magnetization per spin', xscale='log')
    # ax.legend(loc='best')
    # plt.savefig('magnetization.pdf')
    # plt.show()

if __name__ == '__main__':
    # set default font size for axis labels
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12

    # set default legend font size
    mpl.rcParams['legend.fontsize'] = 12

    main()