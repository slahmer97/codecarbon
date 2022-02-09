import pandas as pd

alex = pd.read_csv("dall.csv")
d7 = pd.read_csv("d7.csv")
d7r = pd.read_csv("d7r.csv")


cpuall = (alex["duration"]*alex["cpu"]).mean()
gpuall = (alex["duration"]*alex["gpu"]).mean()
ramall = (alex["duration"]*alex["ram"]).mean()



cpud7 = (d7["duration"]*d7["cpu"]).mean()
gpud7 = (d7["duration"]*d7["gpu"]).mean()
ramd7 = (d7["duration"]*d7["ram"]).mean()

cpud7r = (d7r["duration"]*d7r["cpu"]).mean()
gpud7r = (d7r["duration"]*d7r["gpu"]).mean()
ramd7r = (d7r["duration"]*d7r["ram"]).mean()



print("{} -- {} -- {}".format(cpuall, gpuall, ramall))

print("{} -- {} -- {}".format(cpud7, gpud7, ramd7))

print("{} -- {} -- {}".format(cpud7r+cpud7, gpud7r+gpud7, ramd7r+ramd7))
