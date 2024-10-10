import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

alpha = 1 / 7
beta = 0.3
gamma = 1 / 7
mu = 1 / 50
vac = 7.5 / 10

N = 1000
n_pop = 1000

v0 = 0
d0 = 0
r0 = 0
i0 = 5
e0 = 1 / 5
s0 = n_pop - i0


#import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp






# SIR model
#y0 = s0, i0, r0

# SEIR model
#y0 = s0, e0, i0, r0

# SEIRD model
# y0 = s0, e0, i0, r0, d0

# SEIRD model
#y0 = s0, e0, i0, r0, d0, v0

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

def seird_ode(t, y):
    s, e, i, r, d = y

    s_prime = -beta * (i / n_pop) * s
    e_prime = beta * (i / n_pop) * s - alpha * e
    i_prime = alpha * e - gamma * i - mu * i
    r_prime = gamma * i
    d_prime = mu * i

    return [s_prime, e_prime, i_prime, r_prime, d_prime]

def seirdv_ode(t, y):
    s, e, i, r, d, v = y

    s_prime = -beta * (i / n_pop) * s - vac
    e_prime = beta * (i / n_pop) * s - alpha * e
    i_prime = alpha * e - gamma * i - mu * i
    r_prime = gamma * i
    d_prime = mu * i
    v_prime = vac

    return [s_prime, e_prime, i_prime, r_prime, d_prime, v_prime]

def SSA(prop, stoch, X0, tspan, coeff):

    # prop - propensities
    # stoch - stiochiometry vector
    # Initial state vector
    tvec = np.zeros(1)
    tvec[0] = tspan[0]
    Xarr = np.zeros([1, len(X0)])
    Xarr[0, :] = X0
    t = tvec[0]
    X = X0
    sMat = stoch
    while t < tspan[1]:
        r1, r2 = np.random.uniform(0, 1, size=2)  # Find two random numbers on
        #uniform distr.
        re = prop(X, coeff)
        cre = np.cumsum(re)
        a0 = cre[-1]
        if a0 < 1e-12:
            break
        tau = np.random.exponential(scale=1/a0)  # Random number exponential
        #distribution
        cre = cre/a0
        r = 0
        while cre[r] < r2:
            r += 1
            t += tau
            # if new time is larger than final time, skip last calculation
            if t > tspan[1]:
                break
        tvec = np.append(tvec, t)
        X = X + sMat[r, :]
        Xarr = np.vstack([Xarr, X])
        # If iterations stopped before final time, add final time and no change
    if tvec[-1] < tspan[1]:
            tvec = np.append(tvec, tspan[1])
            Xarr = np.vstack([Xarr, X])
    return tvec, Xarr

def propensities_sir(X, coeff):
    S, I, R = X  # Current state
    beta, gamma, N = coeff  # Model parameters: beta, gamma, N (total population)
    
    # Calculate propensities (rates)
    infection_rate = beta * S * I / N
    recovery_rate = gamma * I
    
    return np.array([infection_rate, recovery_rate])



def propensities_seir(X, coeff):
    S, E, I, R = X  # Current state
    beta, gamma, alpha, N = coeff  # Model parameters: beta, gamma, N (total population)
    # Calculate propensities (rates)
    expotion_rate = beta * S * I / N
    infection_rate = alpha * E- gamma * I
    recovery_rate = gamma * I
    
    
    return np.array([expotion_rate, infection_rate, recovery_rate])

def propensities_seird(X, coeff):
    S, E, I, R, D = X  # Current state
    beta, gamma, alpha, mu, N = coeff  # Model parameters: beta, gamma, N (total population)
    # Calculate propensities (rates)
    exposure_rate = beta * S * I / N
    infection_rate = alpha * E
    recovery_rate = gamma * I
    death_rate = mu * I

    
    return np.array([exposure_rate, infection_rate, recovery_rate, death_rate])

def propensities_seirdv(X, coeff):
    S, E, I, R, D, V = X  # Current state
    beta, gamma, alpha, mu, vac, N = coeff  # Model parameters
    
    # Calculate propensities (reaction rates)
    exposure_rate = beta * S * I / N       # S -> E
    infection_rate = alpha * E             # E -> I
    recovery_rate = gamma * I              # I -> R
    death_rate = mu * I                    # I -> D
    vaccination_rate = vac                 # S -> V
    
    return np.array([exposure_rate, infection_rate, recovery_rate, death_rate, vaccination_rate])
    

# SIR model

# stochiometry = np.array([[-1, 1, 0],  
#                         [0, -1, 1]])

# X0 = np.array([995, 5, 0])

# tspan = [0, 120] 

# coeff = [0.3, 1/7, 1000]
# tvec, Xarr = SSA(propensities_sir, stochiometry, X0, tspan, coeff)


# plt.plot(tvec, Xarr[:, 0], label='Susceptible')
# plt.plot(tvec, Xarr[:, 1], label='Infected')
# plt.plot(tvec, Xarr[:, 2], label='Recovered')



# SEIR model
# stochiometry = np.array([[-1, 1, 0, 0],
#                         [0, -1, 1, 0],  
#                         [0, 0, -1, 1]])

# X0 = np.array([995, 0, 5, 0])

# tspan = [0, 120] 

#coeff = [beta, gamma, alpha, mu, N]
# tvec, Xarr = SSA(propensities_seir, stochiometry, X0, tspan, coeff)


# plt.plot(tvec, Xarr[:, 0], label='Susceptible')
# plt.plot(tvec, Xarr[:, 1], label='Exposed')
# plt.plot(tvec, Xarr[:, 2], label='Infected')
# plt.plot(tvec, Xarr[:, 3], label='Recovered')

# SEIRD model
# stochiometry = np.array([[-1, 1, 0, 0, 0],
#                         [0, -1, 1, 0, 0],  
#                         [0, 0, -1, 1, 0],
#                         [0, 0, -1, 0, 1 ]])

# X0 = np.array([995, 0, 5, 0, 0])

# tspan = [0, 120] 

# coeff = [beta, gamma, alpha, mu, N]
# tvec, Xarr = SSA(propensities_seird, stochiometry, X0, tspan, coeff)


# plt.plot(tvec, Xarr[:, 0], label='Susceptible')
# plt.plot(tvec, Xarr[:, 1], label='Exposed')
# plt.plot(tvec, Xarr[:, 2], label='Infected')
# plt.plot(tvec, Xarr[:, 3], label='Recovered')
# plt.plot(tvec, Xarr[:, 4], label='Dead')

# SEIRDV model
stochiometry = np.array([[-1, 1, 0, 0, 0, 0],
                        [0, -1, 1, 0, 0, 0],  
                        [0, 0, -1, 1, 0, 0],
                        [0, 0, -1, 0, 1, 0],
                        [-1, 0, 0, 0, 0, 1]])

X0 = np.array([995, 0, 5, 0, 0, 0])

tspan = [0, 120] 

coeff = [beta, gamma, alpha, mu, vac, N]
tvec, Xarr = SSA(propensities_seirdv, stochiometry, X0, tspan, coeff)

plt.plot(tvec, Xarr[:, 0], label='Susceptible')
plt.plot(tvec, Xarr[:, 1], label='Exposed')
plt.plot(tvec, Xarr[:, 2], label='Infected')
plt.plot(tvec, Xarr[:, 3], label='Recovered')
plt.plot(tvec, Xarr[:, 4], label='Dead')
plt.plot(tvec, Xarr[:, 5], label='Vaccinated')



plt.title("Projekt 2")
plt.grid()
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.show()