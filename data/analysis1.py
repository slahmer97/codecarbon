import pandas as pd

for i in range(1, 12):
    a = pd.read_csv("alexnet-gpu-tx2/a{}.csv".format(i))
    ar = pd.read_csv("alexnet-gpu-tx2/a{}r.csv".format(i))

    duration1 = a["duration"]
    ram1 = a["ram"]
    cpu1 = a["cpu"]
    gpu1 = a["gpu"]

    duration2 = ar["duration"]
    ram2 = ar["ram"]
    cpu2 = ar["cpu"]
    gpu2 = ar["gpu"]

    ram1 = ((ram1 - 418) * duration1).mean()
    ram2 = ((ram2 - 418) * duration2).mean()

    cpu1 = ((cpu1 - 458) * duration1).mean()
    cpu2 = ((cpu2 - 458) * duration2).mean()

    gpu1 = ((gpu1 - 221) * duration1).mean()
    gpu2 = ((gpu2 - 221) * duration2).mean()

    print("{} -- {} -- {} -- {} -- {} ".format(gpu1 + gpu2, ram1 + ram2, cpu1 + cpu2,
                                               gpu1 + gpu2,
                                               duration1.mean() + duration2.mean()))

alex = pd.read_csv("alexnet-gpu-tx2/a.csv")
duration = alex["duration"]
ram = ((alex["ram"] - 418) * duration).mean()
cpu = ((alex["cpu"] - 458) * duration).mean()
gpu = ((alex["gpu"] - 221) * duration).mean()

print("{} -- {} -- {} -- {} -- {}".format(ram + gpu, ram, cpu, gpu, duration.mean()))
