import numpy as np
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt

g = np.array([0, (-9.82)])
c = 0.05
km = 700
destination = np.array([80, 60])
k_p = 0.93 # Gotten through trial and error

y0 = np.array([0, 0, 0, 0])

def F(m, v):
    f = m*g - c * np.linalg.norm(v) * v
    return f

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
    theta = thetaoptimized(t, pos, vel)
    return np.array([km * np.cos(theta), km * np.sin(theta)], dtype=np.float64)

def v(t):
    if t == 0:
         return np.array([0, 0])
    else: 
        return np.array([10, 0])    

# def theta(t, pos): # Replaced by thetaoptimized
#     if pos[1]<= 20:
#         return math.pi/2
#     else:
#         return np.arctan2(destination[1]-pos[1], destination[0]-pos[0])
    
def thetaoptimized(t, pos, vel):
    if pos[1] <= 20:
        return math.pi/2
    
    target_direction = np.arctan2(destination[1] - pos[1], destination[0] - pos[0])  
    current_direction = np.arctan2(vel[0], vel[1])
    
    angle_diff = target_direction - current_direction
    new_angle = current_direction + k_p * angle_diff
    
    return new_angle

t0 = 0
t1 = 50

tspan = (t0, t1)
tt = np.arange(t0, t1, 0.1)

sol = solve_ivp(oderhs, tspan, y0, t_eval=tt)

plt.plot(sol.y[0], sol.y[1], label='x-position')
plt.plot(destination[0], destination[1], "ro")

plt.ylim(0, 100) # Negative height is not possible and thus irrelevant

min_distance = 10000 # Large number to ensure that the first distance is smaller
print(min_distance) 
for i in range(sol.y[0].size):
    dist = math.sqrt((sol.y[0][i] - destination[0])**2 + (sol.y[1][i] -  destination[1])**2)

    if dist < min_distance:
        min_distance = dist
    
print(min_distance)
# Value for unoptimized distance of target: 8.53166
# With optimized: 0.3967

plt.grid(True)
plt.show()