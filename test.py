from codecarbon import EmissionsTracker
tracker = EmissionsTracker(measure_power_secs=0.1)
tracker.start()

a = []
for i in range(100000000):
    a.append( 10^i)

emissions = tracker.stop()
