#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:32:50 2024

"""
import numpy as np
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt
g = np.array([0, (-9.82)])
c = 0.05
km = 700
stoppos = np.array([80, 60])


y0 = np.array([0, 0, 0, 0])

def oderhs(t, y):
    m_val = m(t)
    x, y, vx, vy = y
    pos = [x, y]
    f = F(m_val, np.array([vx, vy]))
    mp = mprime(t)
    u = U(t, pos)
    dxdt = vx
    dydt = vy
    dvxdt, dvydt =(f + mp*u) /m_val
    
    return [dxdt, dydt, dvxdt, dvydt]


def F(m, v):

    f = m*g - c * np.linalg.norm(v) * v
    return f

def m(t):
    if t <= 10:
        return 8-0.4*t
    else:
        return 4

def mprime(t):
    if t <= 10:
        return 0.4
    else: 
        return 0

def U(t, pos):
    theta = phi(t, pos)
    return np.array([km * np.cos(theta), km * np.sin(theta)], dtype=np.float64)

def phi(t, pos):
        if pos[1]<= 20:
            return math.pi/2
        else:
            return np.arctan2(stoppos[0]-pos[1], stoppos[1]-pos[0])
    
def v(t):
    if t == 0:
         return np.array([0, 0])
    else: 
        return np.array([10, 0])


t_span = (0, 50)
t_eval = np.linspace(0, 50, 1000)

sol = solve_ivp(oderhs, t_span, y0, t_eval=t_eval)

plt.plot(sol.y[0], sol.y[1], label='x-position')
plt.plot(stoppos[1], stoppos[0], "ro")
#plt.plot(sol.t, sol.y[3], label='y-position')

plt.ylim(0, 100)
min_distance = 10000 
for i in range(sol.y[0].size):
    dist = math.sqrt((sol.y[0][i] - stoppos[0])**2 + (sol.y[1][i] -  stoppos[1])**2)
    if dist < min_distance:
        min_distance = dist
    
       
print(min_distance)
#Value for unoptimized distance of target: 8.53166


plt.grid(True)
plt.show()

         
         
    
    
    
    
