import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from itertools import islice
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
    x_bins = np.linspace(min(solution[:, 0]), max(solution[:, 0]), num_bins)
    y_bins = np.linspace(min(solution[:, 1]), max(solution[:, 1]), num_bins)
    x_digitized = np.digitize(solution[:, 0], x_bins)
    y_digitized = np.digitize(solution[:, 1], y_bins)
    # print(both_digitized)

    for block_size in block_size_range:
        # num_blocks = len(solution) // block_size
        # block_entropies = []
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


# Parameters
alpha = 0.3  # Natural growth rate of rabbits
beta = 0.02  # Natural dying rate of rabbits due to predation
gamma = 0.2  # Natural dying rate of foxes
delta = 0.01 # Rate at which predators increase by consuming prey

# Initial number of rabbits and foxes
x0 = 100 #rabbits
y0 = 30  #foxes
z0 = [x0, y0]

# Time points in days
t = np.linspace(0, 200, 1000)

# Integrate the equations over the time grid, t
solution = odeint(model, z0, t, args=(alpha, beta, delta, gamma))

# Plotting
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, solution[:, 0], label='Rabbits (Prey)')
plt.plot(t, solution[:, 1], label='Foxes (Predators)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()

# Example of calculating block entropy for a range of block sizes
num_bins = 10
block_size_range = range(1, 101)
block_entropy_values_range = block_entropy_for_range(solution, block_size_range, num_bins)
t2 = np.linspace(1, 100, 100)

plt.subplot(2, 1, 2)
# plt.figure(figsize = (10, 6))
plt.plot(t2, block_entropy_values_range)
plt.xlabel('Length')
plt.ylabel('Block Entropy')
plt.title('Block Entropy Over Time')

plt.tight_layout()
plt.show()
