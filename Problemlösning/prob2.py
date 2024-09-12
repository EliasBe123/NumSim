#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:32:26 2024

@author: BestElias115
"""

import numpy as np
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt




def ode_rhs(t, x):
    x1, x2 = x
    return [x2, -3*x2-2*x1]



t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)
x = [2, 1]
sol = solve_ivp(ode_rhs, t_span, x, t_eval=t_eval)

plt.plot(sol.t, sol.y[0], label="y(t)")
plt.plot(sol.t, sol.y[1], label="y'(t)")
plt.legend()
plt.show()
