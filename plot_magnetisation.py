import numpy as np
import matplotlib.pyplot as plt


#print(np.sum(np.loadtxt("white.dat") + np.loadtxt("black.dat")))
plt.plot(np.loadtxt("global_market.dat"))
#rng1 = np.loadtxt("rng1.dat")
#plt.plot(np.arange(0, len(rng1)), rng1)
#rng2 = np.loadtxt("rng2.dat")
#plt.plot(rng2)

#print(np.mean(rng1))
#print(np.mean(rng2))
plt.show()
