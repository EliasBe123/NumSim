#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:27:20 2024

"""

import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp



def impliciteuler(f, tspan, u0, dt, *args):
    t1, t2 = tspan
    t_vec = np.arange(t1, t2+1.e-14,dt)
    dt_vec = dt*np.ones_like(t_vec)
    if t_vec[-1] < tspan[1]:
        t_vec = np.append(t_vec,tspan[1])
        dt_vec = np.append(dt_vec, t_vec[-1]-t_vec[-2])
    u = np.zeros((len(t_vec),len(u0)))
    u[0,:] = u0
    for i in range(len(t_vec)-1):
        def g(y):
            return y - u[i, :] - dt_vec[i] * f(t_vec[i+1], y, *args)
        u[i+1, :] = fsolve(g,u[i,:]) 
    return t_vec, u



def ode_rhs(t, y):
    return np.sin(t) - y


t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)
y0 = [0]
sol = RK(ode_rhs, t_span, y0, 0.1)


plt.plot(sol[0], sol[1])


plt.show()