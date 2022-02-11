import pandas as pd
import numpy as np

ram1_means = []
ram2_means = []
cpu1_means = []
cpu2_means = []
gpu1_means = []
gpu2_means = []

total_energy_per = []
total_energy_std = []
for i in range(1, 12):
    tmp = pd.read_csv("a{}.csv".format(i))
    tmpr = pd.read_csv("a{}r.csv".format(i))

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

x_params = [1115232, 0, 614656, 0, 885120, 1327488, 884992,

            4198400, 16781312, 4097000, 1001000]
x_flops = [
]

x = np.log(np.cumsum(x_params, dtype=float))

print(len(x))
# plotting
plt.xlabel('Number of Params (Log-scale)')
plt.ylabel('Normalized Energy')

plt.title('CDF of normalized energy')

plt.plot(x, total_energy_per, marker='*')

plt.savefig("cdf-params-energy-alexnet.pdf")
plt.show()
