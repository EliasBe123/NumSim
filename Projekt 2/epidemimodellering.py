import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

alpha = 1
beta = 0.3
gamma = 1 / 7

n_pop = 1000

r0 = 0
i0 = 5
e0 = 1 / 5
s0 = n_pop - i0

# SIR model
# y0 = s0, i0, r0
# SEIR model
y0 = s0, e0, i0, r0

t0, t1 = 0, 120
tspan = t0, t1

h = 0.1
teval = np.arange(t0, t1, h)

def sir_ode(t, y):
    s, i, r = y

    s_prime = -beta * (i / n_pop) * s
    i_prime = beta * (i / n_pop) * s - gamma * i
    r_prime = gamma * i

    return [s_prime, i_prime, r_prime]

def seir_ode(t, y):
    s, e, i, r = y

    s_prime = -beta * (i / n_pop) * s
    e_prime = beta * (i / n_pop) * s - alpha * e
    i_prime = alpha * e - gamma * i
    r_prime = gamma * i

    return [s_prime, e_prime, i_prime, r_prime]


# SIR model
# y = solve_ivp(sir_ode, tspan, y0, t_eval=teval)
# plt.plot(y.t, y.y[0], label="s(t)")
# plt.plot(y.t, y.y[1], label="i(t)")
# plt.plot(y.t, y.y[2], label="r(t)")

# SEIR model
y = solve_ivp(seir_ode, tspan, y0, t_eval=teval)
plt.plot(y.t, y.y[0], label="s(t)")
plt.plot(y.t, y.y[1], label="e(t)")
plt.plot(y.t, y.y[2], label="i(t)")
plt.plot(y.t, y.y[3], label="r(t)")
plt.title("Projekt 2")
plt.grid()
plt.legend()
plt.show()