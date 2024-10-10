import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

alpha = 1 / 7
beta = 0.3
gamma = 1 / 7
mu = 264 / 100000
vac = 5 / 10

n_pop = 1000

iv0 = 0
v20 = 0
v10 = 0
d0 = 0
r0 = 0
i0 = 5
e0 = 1 / 5
s0 = n_pop - i0

y0 = s0, e0, i0, r0, d0, v10, v20, iv0

t0, t1 = 0, 120
tspan = t0, t1

h = 0.1
teval = np.arange(t0, t1, h)

def own_model_ode(t, y):
    s, e, i, r, d, v1, v2, iv = y

    s_prime = -beta * (i / n_pop) * s - vac
    e_prime = beta * (i / n_pop) * s - alpha * e
    i_prime = alpha * e - gamma * i - mu * i
    r_prime = gamma * i
    d_prime = mu * i
    v1_prime = vac
    v2_prime = vac
    iv_prime = vac

    return [s_prime, e_prime, i_prime, r_prime, d_prime, v1_prime, v2_prime, iv_prime]

y = solve_ivp(own_model_ode, tspan, y0, t_eval=teval)

plt.plot(y.t, y.y[0], label="s(t)")
plt.plot(y.t, y.y[1], label="e(t)")
plt.plot(y.t, y.y[2], label="i(t)")
plt.plot(y.t, y.y[3], label="r(t)")
plt.plot(y.t, y.y[4], label="d(t)")
plt.plot(y.t, y.y[5], label="v1(t)")
plt.plot(y.t, y.y[6], label="v2(t)")
plt.plot(y.t, y.y[7], label="iv(t)")

plt.title("Projekt 2")
plt.grid()
plt.legend()
plt.show()