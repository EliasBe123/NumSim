import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

alpha = 1 / 7
beta = 0.3
gamma = 1 / 7
mu = 1 /21
vac = 7.5 / 10

v1 = 1/10
v2 = 1/28
im = 1/14

N = 1000
RV2_0 = 0
RV1_0 = 0
IMREC_0 = 0
IM_0 = 0
V2_0 = 0
V1_0 = 0
V_0 = 0
D_0 = 0
R_0 = 0
I_0 = 5
E_0 = 0
S_0 = N - I_0

coeff = np.array([beta, gamma, alpha, mu, vac, v1,v2, im, N])


# SIR model
SIR_0 = np.array([S_0, I_0, R_0])

# SEIR model
SEIR_0 = np.array([S_0, E_0, I_0, R_0])

# SEIRD model
SEIRD_0 = np.array([S_0, E_0, I_0, R_0, D_0])

# SEIRDV model
SEIRDV_0 = np.array([S_0, E_0, I_0, R_0, D_0, V_0])

# Own model
SEIRDV1V2IM_0_ode = np.array([S_0, E_0, I_0, R_0, D_0, V1_0, V2_0, IM_0, IMREC_0, RV1_0, RV2_0])
SEIRDV1V2IM_0_stoch = np.array([S_0, E_0, I_0, R_0, D_0, V1_0, V2_0, IM_0])

t0, t1 = 0, 120
tspan = t0, t1

h = 0.1
teval = np.arange(t0, t1, h)

def sir_ode(t, y):
    S, I, R = y

    S_prime = -beta * (I / N) * S
    I_prime = beta * (I / N) * S - gamma * I
    R_prime = gamma * I

    return np.array([S_prime, I_prime, R_prime])

def seir_ode(t, y):
    S, E, I, R = y

    S_prime = -beta * (I / N) * S
    E_prime = beta * (I / N) * S - alpha * E
    I_prime = alpha * E - gamma * I
    R_prime = gamma * I

    return np.array([S_prime, E_prime, I_prime, R_prime])

def seird_ode(t, y):
    S, E, I, R, D = y

    S_prime = -beta * (I / N) * S
    E_prime = beta * (I / N) * S - alpha * E
    I_prime = alpha * E - gamma * I - mu * I
    R_prime = gamma * I
    D_prime = mu * I

    return np.array([S_prime, E_prime, I_prime, R_prime, D_prime])

def seirdv_ode(t, y):
    S, E, I, R, D, V = y

    S_prime = -beta * (I / N) * S - vac
    E_prime = beta * (I / N) * S - alpha * E
    I_prime = alpha * E - gamma * I - mu * I
    R_prime = gamma * I
    D_prime = mu * I
    V_prime = vac

    return np.array([S_prime, E_prime, I_prime, R_prime, D_prime, V_prime])

def own_model_ode(t, y):
    S, E, I, R, D, V1, V2, IM, IM_REC, R_V1, R_V2 = y
    
    exposure_rate = beta * S * I / N
    exposure_rate_v1 = (beta / 2) * V1 * (I / N)
    infection_rate = alpha * E
    recovery_rate = gamma * I
    death_rate = mu * I
    vaccination_rate_v1 = v1 * S
    vaccination_rate_v2 = v2 * V1
    full_immunity_from_vaccination = im * V2
    full_immunity_from_recovery = im * R
    recovery_to_v1 = v1 * R
    recovery_to_v2 = v2 * R

    return np.array([exposure_rate, exposure_rate_v1, infection_rate,
                     recovery_rate, death_rate, vaccination_rate_v1,
                     vaccination_rate_v2, full_immunity_from_vaccination,
                     full_immunity_from_recovery, recovery_to_v1, recovery_to_v2])

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
    beta, gamma, alpha, mu, vac, v1, v2, im, N = coeff  # Model parameters

    # Calculate propensities (rates)
    infection_rate = beta * S * I / N           # S  -> I
    recovery_rate = gamma * I                   # I  -> R
    
    return np.array([infection_rate, recovery_rate])

def propensities_seir(X, coeff):
    S, E, I, R = X  # Current state
    # beta, gamma, alpha, N = coeff  # Model parameters: beta, gamma, N (total population)
    beta, gamma, alpha, mu, vac, v1, v2, im, N = coeff  # Model parameters

    # Calculate propensities (rates)
    expotion_rate = beta * S * I / N            # S  -> E
    infection_rate = alpha * E- gamma * I       # E  -> I
    recovery_rate = gamma * I                   # I  -> R
    
    return np.array([expotion_rate, infection_rate, recovery_rate])

def propensities_seird(X, coeff):
    S, E, I, R, D = X  # Current state
    # beta, gamma, alpha, mu, N = coeff  # Model parameters: beta, gamma, N (total population)
    beta, gamma, alpha, mu, vac, v1, v2, im, N = coeff  # Model parameters

    # Calculate propensities (rates)
    exposure_rate = beta * S * I / N            # S  -> E
    infection_rate = alpha * E                  # E  -> I
    recovery_rate = gamma * I                   # I  -> R
    death_rate = mu * I                         # I  -> D

    return np.array([exposure_rate, infection_rate, recovery_rate, death_rate])

def propensities_seirdv(X, coeff):
    S, E, I, R, D, V = X  # Current state
    # beta, gamma, alpha, mu, vac, N = coeff  # Model parameters
    beta, gamma, alpha, mu, vac, v1, v2, im, N = coeff  # Model parameters
    
    # Calculate propensities (reaction rates)
    exposure_rate = beta * S * I / N            # S  -> E
    infection_rate = alpha * E                  # E  -> I
    recovery_rate = gamma * I                   # I  -> R
    death_rate = mu * I                         # I  -> D
    vaccination_rate = vac                      # S  -> V
    
    return np.array([exposure_rate, infection_rate, recovery_rate, death_rate, vaccination_rate])
    
def propensities_seirdv1v2im(X, coeff):
    S, E, I, R, D, V1, V2, IM = X  # Current state
    beta, gamma, alpha, mu, vac, v1, v2, im, N = coeff  # Model parameters
    
    # Calculate propensities (reaction rates)
    exposure_rate = beta * S * I / N            # S  -> E
    exposure_rate_v1 = (beta / 2) * V1 * I / N  # V1 -> E
    infection_rate = alpha * E                  # E  -> I
    recovery_rate = gamma * I                   # I  -> R
    death_rate = mu * I                         # I  -> D
    vaccination_rate_v1 = v1 * S                # S  -> V1 (First dose of vaccine)
    vaccination_rate_v2 = v2 * V1               # V1 -> V2 (Second dose of vaccine)
    full_immunity_from_vaccination = im * V2    # V2 -> IMV (Full immunity)
    full_immunity_from_recovery = im * R        # R  -> IMR
    recovery_to_v1 = v1 * R                     # R  -> V1
    recovery_to_v2 = v2 * R                     # R  -> V2
    
    return np.array([exposure_rate, exposure_rate_v1, infection_rate,
                     recovery_rate, death_rate, vaccination_rate_v1,
                     vaccination_rate_v2, full_immunity_from_vaccination,
                     full_immunity_from_recovery, recovery_to_v1, recovery_to_v2])


# ODE
ode = -1

# SIR
if ode == 0:
    y = solve_ivp(fun=sir_ode, t_span=tspan, y0=SIR_0, t_eval=teval)

    plt.plot(y.t, y.y[0], label='Susceptible')
    plt.plot(y.t, y.y[1], label='Infected')
    plt.plot(y.t, y.y[2], label='Recovered')

# SEIR
elif ode == 1:
    y = solve_ivp(fun=seir_ode, t_span=tspan, y0=SEIR_0, t_eval=teval)

    plt.plot(y.t, y.y[0], label='Susceptible')
    plt.plot(y.t, y.y[1], label='Exposed')
    plt.plot(y.t, y.y[2], label='Infected')
    plt.plot(y.t, y.y[3], label='Recovered')

# SEIRD
elif ode == 2:
    y = solve_ivp(fun=seird_ode, t_span=tspan, y0=SEIRD_0, t_eval=teval)

    plt.plot(y.t, y.y[0], label='Susceptible')
    plt.plot(y.t, y.y[1], label='Exposed')
    plt.plot(y.t, y.y[2], label='Infected')
    plt.plot(y.t, y.y[3], label='Recovered')
    plt.plot(y.t, y.y[4], label='Dead')

# SEIRDV
elif ode == 3:
    y = solve_ivp(fun=seirdv_ode, t_span=tspan, y0=SEIRDV_0, t_eval=teval)

    plt.plot(y.t, y.y[0], label='Susceptible')
    plt.plot(y.t, y.y[1], label='Exposed')
    plt.plot(y.t, y.y[2], label='Infected')
    plt.plot(y.t, y.y[3], label='Recovered')
    plt.plot(y.t, y.y[4], label='Dead')
    plt.plot(y.t, y.y[5], label='Vaccinated')

# Own (SEIRDV1V2IM)
elif ode == 4:
    y = solve_ivp(fun=own_model_ode, t_span=tspan, y0=SEIRDV1V2IM_0_ode, t_eval=teval)

    plt.plot(y.t, y.y[0], label='Susceptible')
    plt.plot(y.t, y.y[1], label='Exposed')
    plt.plot(y.t, y.y[2], label='Infected')
    plt.plot(y.t, y.y[3], label='Recovered')
    plt.plot(y.t, y.y[4], label='Dead')
    plt.plot(y.t, y.y[5], label='Vaccinated1')
    plt.plot(y.t, y.y[6], label='Vaccinated2')
    plt.plot(y.t, y.y[7], label='Immune')
    plt.plot(y.t, y.y[8], label='Immune recovery')
    plt.plot(y.t, y.y[9], label='Recovery vaccinated 1')
    plt.plot(y.t, y.y[10], label='Recovery vaccinated 2')


stoch = 4

# SIR
if stoch == 0:
    stochiometry = np.array([[-1, 1, 0],  
                             [ 0,-1, 1]])

    tvec, Xarr = SSA(propensities_sir, stochiometry, SIR_0, tspan, coeff)

    plt.plot(tvec, Xarr[:, 0], label='Susceptible')
    plt.plot(tvec, Xarr[:, 1], label='Infected')
    plt.plot(tvec, Xarr[:, 2], label='Recovered')

# SEIR
elif stoch == 1:
    stochiometry = np.array([[-1, 1, 0, 0],
                             [ 0,-1, 1, 0],  
                             [ 0, 0,-1, 1]])

    tvec, Xarr = SSA(propensities_seir, stochiometry, SEIR_0, tspan, coeff)

    plt.plot(tvec, Xarr[:, 0], label='Susceptible')
    plt.plot(tvec, Xarr[:, 1], label='Exposed')
    plt.plot(tvec, Xarr[:, 2], label='Infected')
    plt.plot(tvec, Xarr[:, 3], label='Recovered')

# SEIRD
elif stoch == 2:
    stochiometry = np.array([[-1, 1, 0, 0, 0],
                             [ 0,-1, 1, 0, 0],  
                             [ 0, 0,-1, 1, 0],
                             [ 0, 0,-1, 0, 1]])

    tvec, Xarr = SSA(propensities_seird, stochiometry, SEIRD_0, tspan, coeff)

    plt.plot(tvec, Xarr[:, 0], label='Susceptible')
    plt.plot(tvec, Xarr[:, 1], label='Exposed')
    plt.plot(tvec, Xarr[:, 2], label='Infected')
    plt.plot(tvec, Xarr[:, 3], label='Recovered')
    plt.plot(tvec, Xarr[:, 4], label='Dead')

# SEIRDV
elif stoch == 3:
    stochiometry = np.array([[-1, 1, 0, 0, 0, 0],
                             [ 0,-1, 1, 0, 0, 0],  
                             [ 0, 0,-1, 1, 0, 0],
                             [ 0, 0,-1, 0, 1, 0],
                             [-1, 0, 0, 0, 0, 1]])

    tvec, Xarr = SSA(propensities_seirdv, stochiometry, SEIRDV_0, tspan, coeff)

    plt.plot(tvec, Xarr[:, 0], label='Susceptible')
    plt.plot(tvec, Xarr[:, 1], label='Exposed')
    plt.plot(tvec, Xarr[:, 2], label='Infected')
    plt.plot(tvec, Xarr[:, 3], label='Recovered')
    plt.plot(tvec, Xarr[:, 4], label='Dead')
    plt.plot(tvec, Xarr[:, 5], label='Vaccinated')

# Own (SEIRDV1V2IM)
elif stoch == 4:
    # S E I R D V1 V2 IM
    stochiometry = np.array([[-1, 1, 0, 0, 0, 0, 0, 0],  # S -> E (Infection)
                             [ 0, 1, 0, 0, 0,-1, 0, 0],  # V1 -> E (Reduced infection for V1)
                             [ 0,-1, 1, 0, 0, 0, 0, 0],  # E -> I (Incubation ends)
                             [ 0, 0,-1, 1, 0, 0, 0, 0],  # I -> R (Recovery)
                             [ 0, 0,-1, 0, 1, 0, 0, 0],  # I -> D (Death)
                             [-1, 0, 0, 0, 0, 1, 0, 0],  # S -> V1 (First vaccine dose)
                             [ 0, 0, 0, 0, 0,-1, 1, 0],  # V1 -> V2 (Second vaccine dose)
                             [ 0, 0, 0, 0, 0, 0,-1, 1],  # V2 -> IM (Full immunity from vaccination)
                             [ 0, 0, 0,-1, 0, 0, 0, 1],  # R -> IM (Full immunity after recovery)
                             [ 0, 0, 0,-1, 0, 1, 0, 0],  #R -> V1
                             [ 0, 0, 0,-1, 0, 0, 1, 0]]) #R -> V2

    tvec, Xarr = SSA(propensities_seirdv1v2im, stochiometry, SEIRDV1V2IM_0_stoch, tspan, coeff)

    plt.plot(tvec, Xarr[:, 0], label='Susceptible')
    plt.plot(tvec, Xarr[:, 1], label='Exposed')
    plt.plot(tvec, Xarr[:, 2], label='Infected')
    plt.plot(tvec, Xarr[:, 3], label='Recovered')
    plt.plot(tvec, Xarr[:, 4], label='Dead')
    plt.plot(tvec, Xarr[:, 5], label='Vaccinated1')
    plt.plot(tvec, Xarr[:, 6], label='Vaccinated2')
    plt.plot(tvec, Xarr[:, 7], label='Immune')

# i = 0
# result = 0
# while (i < 1):
#     tvec, Xarr = SSA(propensities_seirdv1v2im, stochiometry, X0, tspan, coeff)
#     i += 1

#     result += Xarr[:][-1][4]

# res = result/1

plt.title("Projekt 2")
plt.grid()
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.show()