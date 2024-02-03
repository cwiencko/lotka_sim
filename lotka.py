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
#def shannon_entropy(x, y):
    total = x + y
    if total == 0:
        return 0
    p_x = x / total if x != 0 else 0
    p_y = y / total if y != 0 else 0
    entropy = 0
    if p_x > 0:
        entropy -= p_x * np.log(p_x)
    if p_y > 0:
        entropy -= p_y * np.log(p_y)
    return entropy
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

def block_entropy(solution, block_size, num_bins):
    num_blocks = len(solution) // block_size
    block_entropy_values = []

    # Digitize or categorize your data
    x_bins = np.linspace(min(solution[:, 0]), max(solution[:, 0]), num_bins)
    y_bins = np.linspace(min(solution[:, 1]), max(solution[:, 1]), num_bins)

    blocks = sliding_window(solution, block_size)
    for i in range(num_blocks):
        #block = solution[i * block_size:(i + 1) * block_size]
        x_digitized = np.digitize(blocks[i][:, 0], x_bins)
        y_digitized = np.digitize(blocks[i][:, 1], y_bins)

        # Calculate frequencies for each block
        x_frequencies = calculate_frequencies(x_digitized)
        y_frequencies = calculate_frequencies(y_digitized)

        # Calculate entropy for each block
        x_entropy = shannon_entropy(x_frequencies, block_size)
        y_entropy = shannon_entropy(y_frequencies, block_size)

        # Combine or store the entropies as needed
        block_entropy_values.append((x_entropy, y_entropy))

    return block_entropy_values

# Function for calculating block entropy for a range of block sizes
def block_entropy_for_range(solution, block_size_range, num_bins):
    block_entropy_values = {}

    # Digitize or categorize your data
    x_bins = np.linspace(min(solution[:, 0]), max(solution[:, 0]), num_bins)
    y_bins = np.linspace(min(solution[:, 1]), max(solution[:, 1]), num_bins)

    for block_size in block_size_range:
        num_blocks = len(solution) // block_size
        block_entropies = []
        #blocks = sliding_window(solution, block_size)

        for blocks in sliding_window(solution, block_size):
            #block = solution[i * block_size:(i + 1) * block_size]
            x_digitized = np.digitize(blocks[:, 0], x_bins)
            y_digitized = np.digitize(blocks[:, 1], y_bins)

            # Calculate frequencies for each block
            x_frequencies = calculate_frequencies(x_digitized)
            y_frequencies = calculate_frequencies(y_digitized)

            # Calculate entropy for each block
            x_entropy = shannon_entropy(x_frequencies, block_size)
            y_entropy = shannon_entropy(y_frequencies, block_size)

            block_entropies.append((x_entropy, y_entropy))

        block_entropy_values[block_size] = block_entropies

    return block_entropy_values


# Parameters
alpha = 0.3  # Natural growth rate of rabbits
beta = 0.02  # Natural dying rate of rabbits due to predation
gamma = 0.2  # Natural dying rate of foxes
delta = 0.01 # Rate at which predators increase by consuming prey

# Initial number of rabbits and foxes
x0 = 40 #rabbits
y0 = 9  #foxes
z0 = [x0, y0]

# Time points in days
t = np.linspace(0, 200, 1000)

# Integrate the equations over the time grid, t
solution = odeint(model, z0, t, args=(alpha, beta, delta, gamma))

# Calculate entropy over time
#entropy_over_time = [shannon_entropy(x, y) for x, y in solution]

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
block_size_range = range(1, 100)
block_entropy_values_range = block_entropy_for_range(solution, block_size_range, num_bins)

plt.subplot(2, 1, 2)
plt.plot(t, block_entropy_values_range)
plt.xlabel('Time')
plt.ylabel('Shannon Entropy')
plt.title('Shannon Entropy Over Time')
#plt.subplot(2, 1, 2)
#plt.plot(t, entropy_over_time)
#plt.xlabel('Time')
#plt.ylabel('Shannon Entropy')
#plt.title('Shannon Entropy Over Time')

plt.tight_layout()
plt.show()


x_bins = np.linspace(min(solution[:, 0]), max(solution[:, 0]), num_bins)
y_bins = np.linspace(min(solution[:, 1]), max(solution[:, 1]), num_bins)
x_digitized = np.digitize(solution[:, 0], x_bins)
y_digitized = np.digitize(solution[:, 1], y_bins)

x_frequencies = calculate_frequencies(x_digitized)
y_frequencies = calculate_frequencies(y_digitized)

x_entropy = shannon_entropy(x_frequencies, len(solution))
y_entropy = shannon_entropy(y_frequencies, len(solution))

# Calculate joint and marginal probabilities
joint_probs, _, _ = np.histogram2d(x_digitized, y_digitized, bins=num_bins)
joint_probs /= joint_probs.sum()
x_probs = np.sum(joint_probs, axis=1)
y_probs = np.sum(joint_probs, axis=0)

# Calculate mutual information
mutual_info = 0
for i in range(num_bins):
    for j in range(num_bins):
        if joint_probs[i, j] > 0:
            mutual_info += joint_probs[i, j] * np.log(joint_probs[i, j] / (x_probs[i] * y_probs[j]))


# Print or plot mutual information
#print("Mutual Information:", mutual_info, "\n")
