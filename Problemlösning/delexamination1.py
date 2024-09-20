#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:16:32 2024

Elias Benjaminsson
"""

import numpy as np
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt

# Uppgift 1:
# x1 = y -> x1' = y'
# x2 = y' -> x2' = y''
# y''= (-cy'-ky)/m
# y'' = (-c*x2-k*x1)/m

c = 0 #struntvärden
k = 0
m = 0
    
def ODE_func(t, x):
    x1, x2 = x
    return [x2, (-c*x2-k*x1)/m]

# Uppgift 2: 
    
a = 1.1 
b = 0.4 
c= 0.4
d = 0.1

y0 = [20, 10]

t_span = (0, 50)
t_eval = np.linspace(0, 50, 1000)

def LV_func(t, y):
    x, y = y 
    return [a*x-b*x*y, (-c)*y+d*x*y]

sol = solve_ivp(LV_func, t_span, y0, t_eval=t_eval)


plt.plot(sol.t, sol.y[1], 'r--', label='Predators')
plt.plot(sol.t, sol.y[0], 'b', label="Prey")
plt.legend()

plt.show()

#%% Uppgift 3

import numpy as np
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt

a = 1.1 
b = 0.4 
c= 0.4
d = 0.1
def MP(f, tspan, u0, dt, *args):
    t1, t2 = tspan
    t_vec = np.arange(t1, t2+1.e-14,dt)
    dt_vec = dt*np.ones_like(t_vec)
    if t_vec[-1] < tspan[1]:
        t_vec = np.append(t_vec,tspan[1])
        dt_vec = np.append(dt_vec, t_vec[-1]-t_vec[-2])
    u = np.zeros((len(t_vec),len(u0)))
    u[0,:] = u0
    for i in range(len(t_vec)-1):
        k1 = np.array(f(t_vec[i], u[i,:], *args))
        k2 = np.array(f(t_vec[i]+dt_vec/2, u[i,:] + (dt_vec[i]/2*k1)))
        u[i+1, :] = u[i, :] + dt_vec[i]*k2
    return t_vec, u

def LV_func(t, y):
    x, y = y 
    return [a*x-b*x*y, (-c)*y+d*x*y]

y0 = [20, 10]

t_span = (0, 50)
sol =MP(LV_func, t_span, y0, 0.1)

plt.plot(sol[0], sol[1])

#%% Uppgift 4:
    
import numpy as np
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve



def implicit_MP(f, tspan, u0, dt, *args):
    t1, t2 = tspan
    t_vec = np.arange(t1, t2+1.e-14,dt)
    dt_vec = dt*np.ones_like(t_vec)
    if t_vec[-1] < tspan[1]:
        t_vec = np.append(t_vec,tspan[1])
        dt_vec = np.append(dt_vec, t_vec[-1]-t_vec[-2])
    u = np.zeros((len(t_vec),len(u0)))
    u[0,:] = u0
    for i in range(len(t_vec) - 1):
        def g(y):
            mid_t = np.array((t_vec[i] + dt_vec[i]) )
            mid_u = np.array((u[i, :] + y) / 2)
            return y - u[i, :] - dt_vec[i] * np.array(f(mid_t, mid_u, *args))

        u[i+1, :] = fsolve(g, u[i, :])  
    return t_vec, u


def ode_func(t, y):
    return np.exp(10*t*np.sin(y))

t_span = (0, 3)
y0 = [0]

sol = implicit_MP(ode_func, t_span, y0, 0.01)
print(sol[1])
plt.plot(sol[0], sol[1])













