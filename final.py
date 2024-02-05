import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

male_percent = .5
# male_percent1 = .7

tau_1 = -26.08*male_percent + 24.6

def get_alpha_L(male_percent):
    return math.log(67_000 * (1-male_percent))/(tau_1 + 1.25 * (male_percent))

dt = .001
years = 1100  # Increased simulation period
time = np.arange(0, years, dt)

A = np.zeros(len(time))
L = np.zeros(len(time))
J = np.zeros(len(time))
P = np.zeros(len(time))
# A1 = np.zeros(len(time))
# L1 = np.zeros(len(time))
# J1 = np.zeros(len(time))
# P1 = np.zeros(len(time))

# Initial populations
A[0], L[0], J[0], P[0] = 10_000, 500_000, 100_000, 1_000_000
# A1[0], L1[0], J1[0], P1[0] = 10_000, 500_000, 100_000, 1_000_000

a = .00038
h = .2

c = 1.9
alpha_prey = .5
alpha_lam = get_alpha_L(male_percent)
# alpha_lam1 = get_alpha_L(male_percent1)

k_juv = 5_000_000
k_prey = 1_000_000

trans = .1  # Transition constant from juvenile to adult

gamma = .05
gamma_2 = .1

def s(y):
    return c/h + (c/h)*y/k_juv

def g(x):
    return alpha_prey - alpha_prey*x/k_prey

def f(P):
    return a * P / (1 + a * h * P)

for t in range(1, len(time)):
    t_tau_1 = max(0, t - int(tau_1 / dt))
    L[t] = L[t-1] + (alpha_lam * A[t-1] - gamma*L[t-1] - alpha_lam * np.exp(-gamma*tau_1)*A[t_tau_1]) * dt
    J[t] = J[t-1] + (J[t-1]*(-s(J[t-1]) + c*f(P[t-1])) + alpha_lam * np.exp(-gamma*tau_1)*A[t_tau_1] - trans*J[t-1]) * dt
    A[t] = A[t-1] + (trans*J[t-1] - A[t-1]*gamma_2) * dt
    P[t] = P[t-1] + (P[t-1]*g(P[t-1]) - J[t-1]*f(P[t-1])) * dt

# for t in range(1, len(time)):
#     t_tau_1 = max(0, t - int(tau_1 / dt))
#     L1[t] = L1[t-1] + (alpha_lam1 * A1[t-1] - gamma*L1[t-1] - alpha_lam1 * np.exp(-gamma*tau_1)*A1[t_tau_1]) * dt
#     J1[t] = J1[t-1] + (J1[t-1]*(-s(J1[t-1]) + c*f(P1[t-1])) + alpha_lam1 * np.exp(-gamma*tau_1)*A1[t_tau_1] - trans*J[t-1]) * dt
#     A1[t] = A1[t-1] + (trans*J1[t-1] - A1[t-1]*gamma_2) * dt
#     P1[t] = P1[t-1] + (P1[t-1]*g(P1[t-1]) - J1[t-1]*f(P1[t-1])) * dt

# Truncate the first 100 years for graphing
time_adjusted = time[int(100/dt):]
L_adjusted = L[int(100/dt):]
J_adjusted = J[int(100/dt):]
P_adjusted = P[int(100/dt):]
A_adjusted = A[int(100/dt):]
# L1_adjusted = L1[int(100/dt):]
# J1_adjusted = J1[int(100/dt):]
# P1_adjusted = P1[int(100/dt):]
# A1_adjusted = A1[int(100/dt):]
with open('male50.pkl', 'wb') as file:
    pickle.dump([L_adjusted, J_adjusted, P_adjusted, A_adjusted], file)


# Plotting
plt.figure(figsize=(16, 10))
plt.subplot(2, 1, 1)
plt.plot(time_adjusted, L_adjusted, label='Larva')
plt.plot(time_adjusted, J_adjusted, label='Juvenile')
plt.plot(time_adjusted, P_adjusted, label='Prey')
plt.plot(time_adjusted, A_adjusted, label='Adult')
plt.title(f'Population Dynamics of Sea Lamprey and Prey Fish Over {years-100} Years\nMale Percent: {male_percent*100}%')
plt.xlabel('Time (Years)')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.tight_layout()

# plt.subplot(2, 1, 2)
# plt.plot(time_adjusted, L1_adjusted, label='Larva')
# plt.plot(time_adjusted, J1_adjusted, label='Juvenile')
# plt.plot(time_adjusted, P1_adjusted, label='Prey')
# plt.plot(time_adjusted, A1_adjusted, label='Adult')
# plt.title(f'Population Dynamics of Sea Lamprey and Prey Fish Over {years-100} Years\nMale Percent: {male_percent*100}%')
# plt.xlabel('Time (Years)')
# plt.ylabel('Population')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

plt.show()