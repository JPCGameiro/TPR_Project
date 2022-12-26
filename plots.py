import numpy as np
import matplotlib.pyplot as plt


data= np.loadtxt("Captures/attacker_d1_obs_features.dat", dtype=int)
plt.plot(data[:,0], data[:,1], 'k')
plt.show()
plt.plot(data[:,0], data[:,3], 'b')
plt.show()