import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

duration1_avg = []
duration2_avg = []
duration1_std = []
duration2_std = []
duration = []

X = []
for i in range(1, 22):
    tmp = pd.read_csv("v{}.csv".format(i))
    tmpr = pd.read_csv("v{}r.csv".format(i))

    avg_duration1 = (tmp["duration"]*1000).mean()
    duration1_avg.append(avg_duration1)

    avg_duration2 = (tmpr["duration"]*1000).mean()
    duration2_avg.append(avg_duration2)

    std_duration1 = (tmp["duration"]*1000).std()
    duration1_std.append(std_duration1)
    std_duration2 = (tmpr["duration"]*1000).std()
    duration2_std.append(std_duration2)

n_groups = 21

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}
import scipy


def mean_confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h




rects1 = plt.bar(index, duration1_avg, bar_width,
                 alpha=bar_width,
                 color='b',
                 yerr=duration1_std,
                 error_kw=error_config,
                 label='Prev')

rects2 = plt.bar(index + bar_width, duration2_avg, bar_width,
                 alpha=bar_width,
                 color='r',
                 yerr=duration2_std,
                 error_kw=error_config,
                 label='Nex')

a = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
plt.xlabel('Stacked')
plt.ylabel('Latency(ms)')
plt.title('Latency-measurments')
plt.xticks(index + bar_width / 2, index+1)
plt.legend()

plt.tight_layout()
plt.savefig("latency-vgg16.pdf")
plt.show()
