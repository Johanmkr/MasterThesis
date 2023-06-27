import numpy as np
import matplotlib.pyplot as plt 


a = np.linspace(0,np.pi, 100)

b = np.cos(a)

plt.figure()
plt.plot(a,b)
plt.show()
