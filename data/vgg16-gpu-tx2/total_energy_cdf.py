import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

ram1_means = []
ram2_means = []
cpu1_means = []
cpu2_means = []
gpu1_means = []
gpu2_means = []

total_energy_per = []
total_energy_std = []
for i in range(1, 22):
    tmp = pd.read_csv("v{}.csv".format(i))
    tmpr = pd.read_csv("v{}r.csv".format(i))

    ram1 = tmp["duration"] * tmp["ram"]
    ram1_means.append(ram1.mean())
    ram2 = tmpr["duration"] * tmpr["ram"]
    ram2_means.append(ram2.mean())

    cpu1 = tmp["duration"] * tmp["cpu"]
    cpu1_means.append(cpu1.mean())
    cpu2 = tmpr["duration"] * tmpr["cpu"]
    cpu2_means.append(cpu2.mean())

    gpu1 = tmp["duration"] * tmp["gpu"]
    gpu2 = tmpr["duration"] * tmpr["gpu"]
    gpu1_means.append(gpu1.mean())
    gpu2_means.append(gpu2.mean())

    first = cpu1.mean() + ram1.mean() + gpu1.mean()
    second = cpu2.mean() + ram2.mean() + gpu2.mean()
    total_energy_per.append(first / (second + first))

import matplotlib.pyplot as plt

x_params = [1792, 36928, 0, 73856, 147584, 0, 295168, 590080, 590080, 0, 1180160, 2359808, 2359808, 0, 2359808, 2359808,
            2359808, 0, 0, 411045888, 16781312]
x_flops = [
    715.68 * 10e6 + 13.25 * 10e6,
    15.26 * 10e9 + 13.25 * 10e6,
    13.19 * 10e6,
    7.6 * 10e9 + 6.6 * 10e6,
    15.20 * 10e9 + 6.6 * 10e6,
    6.54 * 10e6,
    7.53 * 10e9 + 3.27 * 10e6,
    15.06 * 10e9 + 3.27 * 10e6,
    15.06 * 10e9 + 3.27 * 10e6,
    15.06 * 10e9 + 3.27 * 10e6,
    3.21 * 10e6,
    7.4 * 10e6 + 3.27 * 10e6,
    14.8 * 10e9 + 1.61 * 10e6,
    1.61 * 10e6,
    3.7 * 10e9 + 401.41 * 10e3,
    3.7 * 10e9 + 401.41 * 10e3,
    3.7 * 10e9 + 401.41 * 10e3,
    401.41 * 10e3,
    0,
    822.08 * 10e6 + 4.1 * 10e3,
    33.55 * 10e6 + 4.1 * 10e3,
    # 81.92*10e6+ 10e4

]

x = np.log(np.cumsum(x_params, dtype=float))

print(len(x))
# plotting
plt.xlabel('Number of Params (Log-scale)')
plt.ylabel('Normalized Energy')

plt.title('CDF of normalized energy')

plt.plot(x, total_energy_per, marker='*')

plt.savefig("cdf-params-energy-vgg16.pdf")
plt.show()
