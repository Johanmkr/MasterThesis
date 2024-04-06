import numpy as np
import matplotlib.pyplot as plt
import figure as fg

N = 15 # Number of data points
x = np.linspace(-1,1,10*N)
x_data = np.linspace(-1,1,N)




# Create true function
TrueFunction = lambda x: 1 + 2*x + 10*x**2 + 5*x**3
y = TrueFunction(x_data)
# Create noisy data
np.random.seed(1)
y_noisy = y + np.random.normal(0,1,N)

datapoints = (x_data, y_noisy) # Tuple of datapoints

# Make true function
true_graph = (x, TrueFunction(x)) # Tuple of true function

# Good polynomial fit to data
p = np.polyfit(x_data,y_noisy,3)
y_fit = np.polyval(p,x)

# Underfit
p_under = np.polyfit(x_data,y_noisy,1)
y_under = np.polyval(p_under,x)

# Overfit
p_over = np.polyfit(x_data,y_noisy,12)
y_over = np.polyval(p_over,x)


fig, ax = plt.subplots(1,1,figsize=(12,12))

# Plot data
ax.scatter(*datapoints, marker="x", label='Data', facecolor='red', s=100, zorder=10)
# Plot true function
ax.plot(*true_graph, label='True function', linestyle="dashed", color='black')

# Plot fits
ax.plot(x, y_fit, label='Good fit', color='lime')
ax.plot(x, y_under, label='Underfit', color='orange')
ax.plot(x, y_over, label='Overfit', color='blue')

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f(x)$')

ax.legend()


fg.SaveShow(fig, "fitting", save=True, show=True, tight_layout=True)
