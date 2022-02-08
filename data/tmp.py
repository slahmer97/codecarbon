import time
from threading import Timer
import datetime

a = 0


class PowerSampler(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


count = 0
ram_power = 0
cpu_power = 0
gpu_power = 0
wifi_power = 0
board_power = 0


def sampler(name):
    global count, ram_power, cpu_power, board_power
    count += 1
    with open("/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power2_input", 'r') as f:
        ram_power += float(f.read())
    with open("/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power0_input", 'r') as f:
        board_power += float(f.read())
    with open("/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power1_input", 'r') as f:
        cpu_power += float(f.read())


rt = PowerSampler(0.01, sampler, "World")
try:
    time.sleep(5)
    print(count)
    print(board_power / float(count))
    print(cpu_power / float(count))
    print(ram_power / float(count))

finally:
    rt.stop()
