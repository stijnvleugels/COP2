import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

def set_defaults():
    ''' Set default values for matplotlib rcParams. '''
    # set default font size for axis labels
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12

    # set default legend font size
    mpl.rcParams['legend.fontsize'] = 12

def figure_tcor():
    ''' Plot correlation time as a function of temperature. '''
    df = pd.read_csv("simulation_17april.csv")

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
    plt.savefig('correlation_time.pdf')
    plt.show()

def main():
    set_defaults()
    figure_tcor()

if __name__ == '__main__':
    main()
# fig, ax = plt.subplots(4, 1, figsize=(6, 7), sharex=True, constrained_layout=True)

# for i in range(4):
#     ax[i].scatter(df['Temperature'], df.iloc[:,i*2+2])
#     ax[i].errorbar(df['Temperature'], df.iloc[:,i*2+2], yerr=df.iloc[:,i*2+1], fmt='o')

# ax[0].set(ylabel='Absolute magnetization', title='Mean and standard deviation of quantities per spin')
# ax[1].set(ylabel='Energy')
# ax[2].set(ylabel='Magnetic susceptibility')
# ax[3].set(xlabel='Temperature', ylabel='Specific heat')

# plt.show()
