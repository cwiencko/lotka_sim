import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from itertools import islice
from sklearn.metrics import mutual_info_score
import collections

# Lotka-Volterra equations
def model(z, t, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

def calculate_frequencies(data):
    unique, counts = np.unique(data, return_counts=True)
    frequencies = dict(zip(unique, counts))
    return frequencies

# Shannon Entropy
def shannon_entropy(frequencies, total):
    entropy = 0
    for freq in frequencies.values():
        p = freq / total
        if p > 0:
            entropy -= p * np.log(p)
    return entropy

def sliding_window(iterable, n):
    "Collect data into overlapping fixed-length chunks or blocks."
    # sliding_window('ABCDEFG', 4) --> ABCD BCDE CDEF DEFG
    it = iter(iterable)
    window = collections.deque(islice(it, n-1), maxlen=n)
    for x in it:
        window.append(x)
        yield tuple(window)

# Function for calculating block entropy for a range of block sizes
def block_entropy_for_range(solution, block_size_range, num_bins):
    block_entropy_values = []

    # Digitize or categorize your data
    # x_perc = np.percentile(solution[:, 0], list(range(0, 100, num_bins)))
    # x_digitized = np.digitize(solution[:, 0], x_perc)
    
    # y_perc = np.percentile(solution[:, 1], list(range(0, 100, num_bins)))
    # y_digitized = np.digitize(solution[:, 1], y_perc)
    x_bins = np.linspace(min(solution[:, 0]), max(solution[:, 0]), (num_bins+1))
    y_bins = np.linspace(min(solution[:, 1]), max(solution[:, 1]), (num_bins+1))
    x_digitized = np.digitize(solution[:, 0], x_bins)
    y_digitized = np.digitize(solution[:, 1], y_bins)
    # print(both_digitized)

    for block_size in block_size_range:
        both_digitized = zip(x_digitized, y_digitized)
        freqs = {}
        for blocks in sliding_window(both_digitized, block_size):
            if blocks in freqs:
                curr = freqs[blocks] + 1
                freqs[blocks] = curr
            else:
                freqs[blocks] = 1

        entropy = shannon_entropy(freqs, len(solution) - block_size + 1)
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
    # x_perc = np.percentile(solution[:, 0], list(range(0, 100, round(100 / num_bins))))
    # x_digitized = np.digitize(solution[:, 0], x_perc)
    
    # y_perc = np.percentile(solution[:, 1], list(range(0, 100, round(100 / num_bins))))
    # y_digitized = np.digitize(solution[:, 1], y_perc)

    # print(x_perc)
    # print(x_digitized, y_digitized)

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

# Parameters
alpha = 0.15  # Natural growth rate of rabbits
beta = 0.01  # Natural dying rate of rabbits due to predation
gamma = 0.3  # Natural dying rate of foxes
delta = 0.01 # Rate at which predators increase by consuming prey

# Initial number of prey and predators
x0 = 100 #prey
y0 = 20  #predators
z0 = [x0, y0]

# Time points in days
t = np.linspace(0, 2000, 5000)

# Integrate the equations over the time grid, t
solution = odeint(model, z0, t, args=(alpha, beta, delta, gamma))

# Plotting
plt.figure(figsize=(15, 9))
plt.subplot(3, 1, 1)
plt.plot(t, solution[:, 0], label='Rabbits (Prey)')
plt.plot(t, solution[:, 1], label='Foxes (Predators)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()

# Block entropy calculation
num_bins = 5
block_size_range = range(1, 101)
block_entropy_values_range = block_entropy_for_range(solution, block_size_range, num_bins)
t2 = np.linspace(1, 100, 100)

plt.subplot(3, 1, 2)
plt.plot(t2, block_entropy_values_range)
plt.xlabel('Length')
plt.ylabel('Block Entropy')
plt.title('Block Entropy Over Length')

# Mutual information calculation
time_range = range(-50, 51)
mutual_score = mutual_information(solution, time_range, num_bins)
t3 = np.linspace(-50, 51, 101)

plt.subplot(3, 1, 3)
plt.plot(t3, mutual_score)
plt.xlabel('Time step (t)')
plt.ylabel('Mutual Information')
plt.title('Mutual Information vs Time step')

plt.tight_layout()
plt.show()
