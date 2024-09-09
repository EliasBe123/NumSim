#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:32:50 2024

"""
import numpy as np
import scipy as sp
import math
g = np.arrray(9.82, 0)
c = 0.05
km = 700

def oderhs(F, mprime, U):
    x = F + mprime*U
    return x

def F(m, v):
    f = m*g - c*abs(v)*v
    return f

def m(t):
    if t <= 10:
        return 8-0.4*t
    else:
        return 4

def u(t, phi):
    return np.array(km*math.cos(phi(t)), km*math.sin(phi(t)))

def phi(t):
        return 0
    
def v(t):
    if t == 0:
        return np.array(0, 0)
    else: 
        
    
    
    
    
