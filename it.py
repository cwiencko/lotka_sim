import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from sklearn.metrics import mutual_info_score
import collections
import pickle

def shannon_entropy(frequencies, total):
    entropy = 0
    for freq in frequencies.values():
        p = freq / total
        if p > 0:
            entropy -= p * np.log(p)
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

def block_entropy_for_range(data, block_size_range, bins):
    block_entropy_values = []

    # bins = np.linspace(min(data[:]), max(data[:]), (num_bins+1))
    digitized = np.digitize(data[:], bins)
    # print(both_digitized)
    # new_size = [x * steps for x in block_size_range]
    # print(new_size)
    for block_size in block_size_range:
        # freqs = {}
        # # blocks = sliding_window(digitized, block_size)
        # # freqs = calculate_frequencies(blocks)
        # for blocks in sliding_window(digitized, block_size):
        #     if blocks in freqs:
        #         curr = freqs[blocks] + 1
        #         freqs[blocks] = curr
        #     else:
        #         freqs[blocks] = 1
        freqs = collections.Counter(sliding_window(digitized, block_size))

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

def mutual_information(in1, in2, time_range, num_bins):
    x_bins = np.linspace(min(in1[:]), max(in1[:]), (num_bins+1))
    y_bins = np.linspace(min(in2[:]), max(in2[:]), (num_bins+1))
    x_digitized = np.digitize(in1[:], x_bins)
    y_digitized = np.digitize(in2[:], y_bins)
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

with open('male50.pkl', 'rb') as file:
    data5 = pickle.load(file)

with open('male70.pkl', 'rb') as file:
    data7 = pickle.load(file)

dt = .001
years = 1100  # Increased simulation period
time = np.arange(0, years, dt)
time_adjusted = time[int(100/dt):]

L5 = data5[0]
J5 = data5[1]
P5 = data5[2]
A5 = data5[3]
L7 = data7[0]
J7 = data7[1]
P7 = data7[2]
A7 = data7[3]

sum5 = [sum(values) for values in zip(L5, J5, A5, P5)]
sum7 = [sum(values) for values in zip(L7, J7, A7, P7)]

size = 50000
cL5 = downsample_data(L5, size)
cJ5 = downsample_data(J5, size)
cA5 = downsample_data(A5, size)
cP5 = downsample_data(P5, size)
cL7 = downsample_data(L7, size)
cJ7 = downsample_data(J7, size)
cA7 = downsample_data(A7, size)
cP7 = downsample_data(P7, size)
cSum5 = [sum(values) for values in zip(cL5, cJ5, cA5, cP5)]
cSum7 = [sum(values) for values in zip(cL7, cJ7, cA7, cP7)]

num_bins = 40
bins = np.linspace(min(min(sum5[:], sum7[:])), max(max(sum5[:], sum7[:])), (num_bins+1))
block_size_range = range(1, 1001)
# eL = block_entropy_for_range(cL, block_size_range, num_bins)
# eJ = block_entropy_for_range(cJ, block_size_range, num_bins)
# eA = block_entropy_for_range(cA, block_size_range, num_bins)
# eP = block_entropy_for_range(cP, block_size_range, num_bins)
# eSum5 = block_entropy_for_range(cSum5, block_size_range, bins)
# eSum7 = block_entropy_for_range(cSum7, block_size_range, bins)

time_range = range(-2000, 2001)
mLP5 = mutual_information(cL5,cP5, time_range, num_bins)
mJP5 = mutual_information(cJ5,cP5, time_range, num_bins)
mAP5 = mutual_information(cA5,cP5, time_range, num_bins)
mLJ5 = mutual_information(cL5,cJ5, time_range, num_bins)
mLA5 = mutual_information(cL5,cA5, time_range, num_bins)
mJA5 = mutual_information(cJ5,cA5, time_range, num_bins)

mLP7 = mutual_information(cL7,cP7, time_range, num_bins)
mJP7 = mutual_information(cJ7,cP7, time_range, num_bins)
mAP7 = mutual_information(cA7,cP7, time_range, num_bins)
mLJ7 = mutual_information(cL7,cJ7, time_range, num_bins)
mLA7 = mutual_information(cL7,cA7, time_range, num_bins)
mJA7 = mutual_information(cJ7,cA7, time_range, num_bins)




# with open('entropy.pkl', 'wb') as file:
#     pickle.dump([eSum5, eSum7], file)

# hu5 = eSum5[999] - eSum5[998]
# hu7 = eSum7[999] - eSum7[998]
# excess5 = eSum5[999] - 999 * hu5
# excess7 = eSum7[999] - 999 * hu7

# print("\n hu of 50%: ", hu5)
# print("\n Ending entropy of 50%: ", eSum5[999])
# print("\n Excess entropy of 50%: ", excess5)
# print("\n hu of 70%: ", hu7)
# print("\n Ending entropy of 70%: ", eSum7[999])
# print("\n Excess entropy of 70%: ", excess7)

# plt.figure(figsize=(16, 10))
# plt.subplot(2, 1, 1)
# plt.plot(time_adjusted, L5, label='Larva')
# plt.plot(time_adjusted, J5, label='Juvenile')
# plt.plot(time_adjusted, P5, label='Prey')
# plt.plot(time_adjusted, A5, label='Adult')
# # plt.plot(time_adjusted, sum5, label='50% Sum')
# # plt.plot(time_adjusted, sum7, label='70% Sum')
# plt.title(f'Population Dynamics of Sea Lamprey and Prey Fish Over {years-100} Years\nMale Percent: {0.5*100}%')
# plt.xlabel('Time (Years)')
# plt.ylabel('Population')
# plt.legend()
# plt.grid(True)

# block_size_range = range(1, 1001)
# plt.subplot(2, 1, 2)
# plt.plot(block_size_range, eSum5, label='Entropy of 50%')
# plt.plot(block_size_range, eSum7, label='Entropy of 70%')
# plt.legend()
# plt.grid(True)

plt.figure(figsize=(16, 10))
plt.subplot(2, 1, 1)
plt.plot(time_range, mLP5, label='Mutual information between Larva and Prey')
plt.plot(time_range, mAP5, label='Mutual information between Adults and Prey')
plt.plot(time_range, mJP5, label='Mutual information between Juveniles and Prey')
plt.plot(time_range, mLA5, label='Mutual information between Larva and Adults')
plt.plot(time_range, mLJ5, label='Mutual information between Larva and Juveniles')
plt.plot(time_range, mJA5, label='Mutual information between Juveniles and Adults')
plt.title('50% Male')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_range, mLP7, label='Mutual information between Larva and Prey')
plt.plot(time_range, mAP7, label='Mutual information between Adults and Prey')
plt.plot(time_range, mJP7, label='Mutual information between Juveniles and Prey')
plt.plot(time_range, mLA7, label='Mutual information between Larva and Adults')
plt.plot(time_range, mLJ7, label='Mutual information between Larva and Juveniles')
plt.plot(time_range, mJA7, label='Mutual information between Juveniles and Adults')
plt.title('70% Male')
plt.legend()

plt.tight_layout()
plt.show()