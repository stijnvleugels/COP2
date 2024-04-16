import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("simulation.csv")

fig, ax = plt.subplots(1, 1)
ax.scatter(df['Temperature'], df.iloc[:,1])
ax.set(xlabel='Temperature', ylabel='Correlation time')

fig, ax = plt.subplots(4, 1, figsize=(6, 7), sharex=True, constrained_layout=True)

for i in range(4):
    ax[i].scatter(df['Temperature'], df.iloc[:,i*2+2])
    ax[i].errorbar(df['Temperature'], df.iloc[:,i*2+2], yerr=df.iloc[:,i*2+1], fmt='o')

ax[0].set(ylabel='Absolute magnetization', title='Mean and standard deviation of quantities per spin')
ax[1].set(ylabel='Energy')
ax[2].set(ylabel='Magnetic susceptibility')
ax[3].set(xlabel='Temperature', ylabel='Specific heat')

plt.show()
