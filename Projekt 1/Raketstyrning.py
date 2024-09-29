#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:32:50 2024

"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
g = np.array([0, (-9.81)])
c = 0.05
km = 700
stoppos = np.array([80, 60])
k_p = 0.925 # Gotten through trial and error

y0 = np.array([0, 0, 0, 0])

def oderhs(t, y):
    m_val = m(t)
    x, y, vx, vy = y
    pos = [x, y]
    vel = np.array([vx, vy])
    f = F(m_val, vel)
    mp = mprime(t)
    u = U(t, pos, vel)
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

def U(t, pos, vel):
    theta = thetaopt(t, pos, vel)
    return np.array([km * np.cos(theta), km * np.sin(theta)])

def theta(t, pos):
        if pos[1]<= 20:
            return np.pi/2
        else:
            return np.arctan2(stoppos[1]-pos[1], stoppos[0]-pos[0])
    
def v(t):
    if t == 0:
         return np.array([0, 0])
    else: 
        return np.array([10, 0])
    
#Optimized trajectory, get direction to target and current direction,
#check the difference and change angle by difference times constant
# k_p (experimented to find).
def thetaopt(t, pos, vel):
    if pos[1] <= 20:
        return np.pi/2
    target_direction = np.arctan2(stoppos[1] - pos[1], stoppos[0] - pos[0])  
    current_direction = np.arctan2(vel[0], vel[1])
    
    angle_diff = target_direction - current_direction
    new_angle = current_direction + k_p * angle_diff
    
    return new_angle
    
t_span = (0, 50)
t_eval = np.linspace(0, 50, 1000)
sol = solve_ivp(oderhs, t_span, y0, t_eval=t_eval)


plt.ylim(0, 100)
min_distance = -1 # Negative length is impossible, used to set if first distance
for i in range(sol.y[0].size):
    dist = np.sqrt((sol.y[0][i] - stoppos[0])**2 + (sol.y[1][i] -  stoppos[1])**2)
    if dist < min_distance or min_distance < 0:
        min_distance = dist
    
print("Minimum Distance with solve_ivp(optimized trajectory): ", min_distance)
#Value for unoptimized distance of target:6.383
#With optimized: 0.339


# RUNGE-KUTTA 4:
def RK4(f, tspan, u0, dt, *args):
    t1, t2 = tspan #Find start of and end of span
    t_vec = np.arange(t1, t2+1.e-14,dt) #Create vector for t 
    dt_vec = dt*np.ones_like(t_vec) #Create vector for steplength, so we can add one to the end
    if t_vec[-1] < tspan[1]: 
        t_vec = np.append(t_vec,tspan[1])
        dt_vec = np.append(dt_vec, t_vec[-1]-t_vec[-2])
    u = np.zeros((len(t_vec),len(u0))) #create array of 0's for solution
    u[0,:] = u0
    for i in range(len(t_vec) - 1): #RK algorithm
        k1 = np.array(f(t_vec[i], u[i, :], *args))
        k2 = np.array(f(t_vec[i] + dt_vec[i] / 2, u[i, :] + dt_vec[i] / 2 * k1, *args))
        k3 = np.array(f(t_vec[i] + dt_vec[i] / 2, u[i, :] + dt_vec[i] / 2 * k2, *args))
        k4 = np.array(f(t_vec[i] + dt_vec[i], u[i, :] + dt_vec[i] * k3, *args))
        u[i+1, :] = u[i, :] + (dt_vec[i] / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        #Return t, x pos and y pos of rocket
    return t_vec, u[:, 0], u[:, 1]

RK_t, RK_x, RK_y = RK4(oderhs, t_span, y0, 0.01)

#Check efficiency of trajectory through minimum distance between target and rocket
min_distance = -1 # Negative length is impossible, used to set if first distance
for i in range(RK_x.size):
    dist = np.sqrt((RK_x[i] - stoppos[0])**2 + (RK_y[i] -  stoppos[1])**2)
    if dist < min_distance or min_distance < 0:
        min_distance = dist

print("Minimum Ddistance with own RK4(optimized trajectory)", min_distance)

plt.plot(sol.y[0], sol.y[1], label='solve_ivp (RK45)')
plt.plot(RK_x, RK_y, '--', label="RK4 solution")
plt.plot(stoppos[0], stoppos[1], "ro", label="Target")
plt.legend()
plt.grid()
plt.show()