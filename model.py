import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from sklearn.metrics import mutual_info_score
import collections

def shannon_entropy(frequencies, total):
    entropy = 0
    for freq in frequencies.values():
        p = freq / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

def calculate_frequencies(data):
    unique, counts = np.unique(data, return_counts=True)
    frequencies = dict(zip(unique, counts))
    return frequencies

def downsample_data(data, target_size):
    step = len(data) // target_size
    return data[::step]

def sliding_window(iterable, n):
    "Collect data into overlapping fixed-length chunks or blocks."
    # sliding_window('ABCDEFG', 4) --> ABCD BCDE CDEF DEFG
    it = iter(iterable)
    window = collections.deque(islice(it, n-1), maxlen=n)
    for x in it:
        window.append(x)
        yield tuple(window)

def block_entropy_for_range(data, block_size_range, num_bins):
    block_entropy_values = []

    bins = np.linspace(min(data[:]), max(data[:]), (num_bins+1))
    digitized = np.digitize(data[:], bins)
    # print(both_digitized)
    # new_size = [x * steps for x in block_size_range]
    # print(new_size)
    for block_size in block_size_range:
        freqs = {}
        # blocks = sliding_window(digitized, block_size)
        # freqs = calculate_frequencies(blocks)
        for blocks in sliding_window(digitized, block_size):
            if blocks in freqs:
                curr = freqs[blocks] + 1
                freqs[blocks] = curr
            else:
                freqs[blocks] = 1

        entropy = shannon_entropy(freqs, len(data) - block_size + 1)
        block_entropy_values.append(entropy)

    return block_entropy_values

def shift_series(data, time_diff):
    if time_diff > 0:
        # Shift data to the right, forwards in time
        return np.append(data[time_diff:], np.full(time_diff, np.nan))
    elif time_diff < 0:
        # Shift data to the left, backwards in time
        return np.append(np.full(-time_diff, np.nan), data[:time_diff])
    return data

def mutual_information(solution, time_range, num_bins):
    x_bins = np.linspace(min(solution[:, 0]), max(solution[:, 0]), (num_bins+1))
    y_bins = np.linspace(min(solution[:, 1]), max(solution[:, 1]), (num_bins+1))
    x_digitized = np.digitize(solution[:, 0], x_bins)
    y_digitized = np.digitize(solution[:, 1], y_bins)
    mutual_info = []
    for lag in time_range:
        y_lag = shift_series(y_digitized, lag)

        valid_indices = ~np.isnan(y_lag)
        y_lag = y_lag[valid_indices]
        x_fixed = x_digitized[valid_indices]

        mutual_score = mutual_info_score(x_fixed, y_lag)
        # print(x_fixed, y_lag)
        # mutual_info.append(mutual_info_score(x_fixed, y_lag))
        mutual_info.append(mutual_score)
        # print(mutual_score)
    return mutual_info



# tau_2 = 10 # 10 years
# tau_2 = 10 # 10 years
tau_1 = 10 # 10 years
tau_2 = 1.25 # 1.25 years

# Updated parameters
# gamma_1, gamma_2, gamma_3 = 0.35, 0.55, 0.2 # Assuming gamma_3 is set reasonably
# gamma_1, gamma_2, gamma_3 = 0.35, 0.55, 0.3 # Assuming gamma_3 is set reasonably
# gamma_1, gamma_2, gamma_3 = 0.35, .96, 1e-5 # Assuming gamma_3 is set reasonably
gamma_1, gamma_2, gamma_3 = 0.35, 1.1, 1e-5 # Assuming gamma_3 is set reasonably
# gamma_1, gamma_2, gamma_3 = 0.35, 0.55, 0.2 # Assuming gamma_3 is set reasonably
# alpha_L, alpha_P = 0.94, 1.31
# alpha_L, alpha_P = 5, 1.1
# alpha_L, alpha_P = 1, 3
alpha_L, alpha_P = 1, 10
# a, h, K = .0021, 2, 1_000_000 
# a, h, K = .000021, .7, 1_000_000 
a, h, K = .0021, .7, 1_000_000 

# Time step and simulation time period
dt = .001  # Time step in years
time = np.arange(0, 100, dt)  # Simulate for 100 years

# Initial populations
L = np.zeros(len(time))
J = np.zeros(len(time))
A = np.zeros(len(time))
P = np.zeros(len(time))

# L[0], J[0], A[0], P[0] = 500_000, 12_000, 100_000, 800_000
L[0], J[0], A[0], P[0] = 500_000, 12_000, 10000, 800_000

# Function to calculate f(P)
def f(P):
    return a * P / (1 + a * h * P)

# Euler integration for population dynamics
for t in range(1, len(time)):
    t_tau_1 = max(0, t - int(tau_1 / dt))
    # t_tau_2 = max(0, t - int((tau_1 + tau_2) / dt))
    t_tau_2 = max(0, t - int(tau_2 / dt))
    
    new_larva = alpha_L * A[t-1]
    # new_juvenile = alpha_L * np.exp(-gamma_1 * tau_1) * A[t_tau_1]
    new_juvenile = alpha_L * np.exp(-gamma_1 * tau_1) * A[t_tau_1]
    # new_adult = alpha_L * np.exp(-gamma_1 * tau_1 - gamma_2 * tau_2) * A[t_tau_2]
    new_adult = alpha_L * np.exp(- gamma_2 * tau_2) * J[t_tau_2]

    # MAGIC_CONSTANT = .01
    L[t] = L[t-1] + (new_larva - gamma_1 * L[t-1] - new_juvenile) * dt
    J[t] = J[t-1] + (new_juvenile - J[t-1] * (gamma_2 - f(P[t-1])) - new_adult) * dt
    A[t] = A[t-1] + (new_adult - A[t-1]**2 * gamma_3) * dt
    # A[t] = A[t-1] + (new_adult -A[t-1] * gamma_3) * dt
    P[t] = P[t-1] + (alpha_P * P[t-1] * (1 - P[t-1] / K) - J[t-1] * f(P[t-1])) * dt

    P[t] = max(P[t], 1)
    L[t] = max(L[t], 1)
    J[t] = max(J[t], 1)
    A[t] = max(A[t], 1)

# Plotting

plt.figure(figsize=(15, 9))
plt.subplot(2, 1, 1)
plt.plot(time, L, label='Larva')
plt.plot(time, J, label='Juvenile')
plt.plot(time, A, label='Adult')
plt.plot(time, P, label='Prey')
plt.title('Population Dynamics of Sea Lamprey and Prey Fish Over 100 Years')
plt.xlabel('Time (Years)')
plt.ylabel('Population')
plt.legend()
plt.grid(True)


cL = downsample_data(L, 1000)
cJ = downsample_data(J, 1000)
cA = downsample_data(A, 1000)
cP = downsample_data(P, 1000)
sum_list = [sum(values) for values in zip(cL, cJ, cA, cP)]

num_bins = 20
block_size_range = range(1, 901)
eL = block_entropy_for_range(cL, block_size_range, num_bins)
eJ = block_entropy_for_range(cJ, block_size_range, num_bins)
eA = block_entropy_for_range(cA, block_size_range, num_bins)
eP = block_entropy_for_range(cP, block_size_range, num_bins)
eSum1 = block_entropy_for_range(sum_list, block_size_range, num_bins)
print(eSum1[499])

plt.subplot(2, 1, 2)
plt.plot(block_size_range, eL, label='Larva')
plt.plot(block_size_range, eJ, label='Juvenile')
plt.plot(block_size_range, eA, label='Adult')
plt.plot(block_size_range, eP, label='Prey')
plt.plot(block_size_range, eSum1, label='Sum')
plt.title('Entropy of Populations over block lengths')
plt.xlabel('Block Length (L)')
plt.ylabel('Entropy')
plt.legend()
plt.grid(True)

plt.show()