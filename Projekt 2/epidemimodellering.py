import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

alpha = 1
beta = 0.3
gamma = 1/7

n_pop = 1000
n_infected = 5

t0, t1 = 0, 120
t_span = t0, t1



plt.grid()
plt.title("Projekt 2")
plt.legend()
plt.show()