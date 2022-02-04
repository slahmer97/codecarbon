from codecarbon import EmissionsTracker
tracker = EmissionsTracker()
tracker.start()

a = 1
for i in range(100):
    a *= 10



emissions = tracker.stop()
