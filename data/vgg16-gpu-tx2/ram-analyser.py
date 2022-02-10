import pandas as pd

alex = pd.read_csv("all.csv")
ram1 = []
cpu1 = []
ram2 = []
cpu2 = []
gpu1 = []
gpu2 = []
duration = []

X = []
for i in range(0, 22):
    tmp = pd.read_csv("a{}.csv".format(i))
    tmpr = pd.read_csv("a{}r.csv".format(i))
    avg_total_duration = (tmp["duration"] + tmpr["duration"]).mean()
    avg_ram1 = (tmp["duration"] * (tmp["ram"]-400)).mean()
    avg_ram2 = (tmpr["duration"] * (tmpr["ram"]-400)).mean()

    avg_cpu1 = (tmp["duration"] * (tmp["cpu"] - 450)).mean()
    avg_cpu2 = (tmpr["duration"] * (tmpr["cpu"] - 450)).mean()

    avg_gpu1 = (tmp["duration"] * (tmp["gpu"] - 150)).mean() + avg_ram1 + avg_cpu1
    avg_gpu2 = (tmpr["duration"] * (tmpr["gpu"] - 150)).mean() + avg_ram2 + avg_cpu2

    ram1.append(avg_ram1)
    cpu1.append(avg_cpu1)
    gpu1.append(avg_gpu1)

    ram2.append(avg_ram2)
    cpu2.append(avg_cpu2)
    gpu2.append(avg_gpu2)

    duration.append(avg_total_duration)
    X.append(i)

all_ram = (alex["duration"] * (alex["ram"] - 400)).mean() + (alex["duration"] * (alex["gpu"] - 150)).mean() + (alex["duration"] * (alex["cpu"] - 450)).mean()
print(all_ram)
from plotly import graph_objects as go
data = {
    "model_1": gpu1,
    "model_2": gpu2,
    "labels": X
}

fig = go.Figure(
    data=[
        go.Bar(
            name="RAM-1",
            x=data["labels"],
            y=data["model_1"],
            offsetgroup=1,
        ),
        go.Bar(
            name="RAM-2",
            x=data["labels"],
            y=data["model_2"],
            offsetgroup=1,
            base=data["model_1"],
        ),
    ],
    layout=go.Layout(
        title="Issue Types - Original and Models",
        yaxis_title="Number of Issues"
    )
)
fig.add_hline(all_ram, col="red")

fig.show()
""""


import numpy as np
import matplotlib.pyplot as plt


x = np.arange(len(cpu1))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width - width / 2, ram1, width, label='ram-1')
rects2 = ax.bar(x - width / 2, ram2, width, label='ram-2')
rects3 = ax.bar(x + width / 2, cpu1, width, label='cpu-1')
rects4 = ax.bar(x + width + width / 2, cpu2, width, label='cpu-2')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Energy')
ax.set_title('Wow')
ax.set_xticks(x, x+1)
ax.legend()



fig.tight_layout()

plt.show()
"""
