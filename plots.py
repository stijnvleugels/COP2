import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import sys

""" 
With this file, you can produce the plots for the correlation time and the four quantities of interest, 
after running the simulation with isingmodel.py 
The file expects the name of the csv file as an argument (excluding extension).
"""

def set_defaults():
    ''' Set default values for matplotlib rcParams. '''
    # set default font size for axis labels
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12

    # set default legend font size
    mpl.rcParams['legend.fontsize'] = 12

def figure_tcor(filename:str):
    ''' Plot correlation time as a function of temperature. '''
    df = pd.read_csv(filename)

    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    correlation_times = df['Correlation_time'] / 50**2
    # combine every 3 points into 1 scatter point and plot error bars
    for i in range(0, len(correlation_times), 3):
        yerr_down = np.min(correlation_times[i:i+3])
        yerr_up = np.max(correlation_times[i:i+3])
        yerr_mid = np.median(correlation_times[i:i+3])
        yerr = np.array([[yerr_mid - yerr_down], [yerr_up - yerr_mid]])
        ax.errorbar(df['Temperature'][i], yerr_mid, yerr=yerr, 
                    fmt='o', capsize=3, capthick=1, elinewidth=1, 
                    ecolor='black', color='black', markersize=5)
    ax.axvline(2.269, color='red', linestyle='--')
    ax.set(xlabel='Temperature [J/k$_B$]', ylabel=r'$\tau$ [1/N$^2$]')
    ax.grid(alpha=0.5)
    ax.set_yscale('log')
    ax.text(2.3, 1.5e0, r'T$_{crit}$', color='red', fontsize=14)
    plt.savefig('correlation_time_h1.pdf')
    plt.show()

def figure_quantities(filename:str):
    ''' Plot the four quantities of interest. '''
    df = pd.read_csv(filename)

    fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True, constrained_layout=True)

    for i in range(4):
        for j in range(0, len(df['Temperature']), 3):
            ax[i].errorbar(df['Temperature'][j], df.iloc[j,i*2+2], yerr=df.iloc[j,i*2+3],
                            fmt='o', capsize=5, capthick=1, elinewidth=1, 
                            ecolor='black', color='black', markersize=5)
            ax[i].scatter(df['Temperature'][j], df.iloc[j+1,i*2+2], alpha=0.5, color='blue')
            ax[i].scatter(df['Temperature'][j], df.iloc[j+2,i*2+2], alpha=0.5, color='blue')
        
        ax[i].axvline(2.269, color='red', linestyle='--', label='$T_{crit}$')
        ax[i].grid(alpha=0.5)

    ax[0].set(ylabel='$|m|$')
    ax[1].set(ylabel='$e$')
    ax[2].set(ylabel='$\chi_M$')
    ax[3].set(xlabel='Temperature [J/k$_B$]', ylabel='$C$')
    ax[0].legend(loc='upper right')

    plt.savefig('quantities_h1.pdf')
    plt.show()

def main():
    filename = 'simulation_h1'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    filename = filename + '.csv'

    set_defaults()
    figure_tcor(filename)
    figure_quantities(filename)

if __name__ == '__main__':
    main()