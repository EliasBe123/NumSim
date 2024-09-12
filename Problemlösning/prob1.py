#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:21:07 2024

"""

import numpy as np
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt

#y′(t) + y(t) − sin(t) = 0, y(0) = 0

# --> y'(t) = sin(t)-y(t)
# y0 = 0


def ode_rhs(t, y):
    return np.sin(t) - y


t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)
y0 = [0]
sol = solve_ivp(ode_rhs, t_span, y0, t_eval=t_eval)

print(sol)
plt.plot(sol.t, sol.y[0])

plt.show()

