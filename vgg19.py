import time
from threading import Timer
import torch
import torch.nn as nn
import threading as th

mutex = th.Lock()
count = 0
ram_power = 0
cpu_power = 0
gpu_power = 0
wifi_power = 0
board_power = 0


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


def sampler(name):
    global count, ram_power, cpu_power, board_power, gpu_power, mutex
    mutex.acquire()
    count += 1
    with open("/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power2_input", 'r') as f:
        ram_power += float(f.read())
    with open("/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power0_input", 'r') as f:
        board_power += float(f.read())
    with open("/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power1_input", 'r') as f:
        cpu_power += float(f.read())
    with open("/sys/bus/i2c/drivers/ina3221x/0-0040/iio_device/in_power0_input", 'r') as f:
        gpu_power += float(f.read())
    mutex.release()


def reset():
    global count, ram_power, cpu_power, board_power, gpu_power, mutex
    count = ram_power = cpu_power = board_power = gpu_power = 0


class VGG(nn.Module):
    def __init__(
        self, num_classes: int = 1000) -> None:
        super().__init__()

        #    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.features3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        self.features4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        )

        self.features5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        self.features6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        self.features7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        self.features8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.features9 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        self.features10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        self.features11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        self.features12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.features13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        self.features14 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        self.features15 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        self.features16 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True)
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = self.features6(x)
        x = self.features7(x)
        x = self.features8(x)
        x = self.features9(x)
        x = self.features10(x)
        x = self.features11(x)
        x = self.features12(x)
        x = self.features13(x)
        x = self.features14(x)
        x = self.features15(x)
        x = self.features16(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        x = self.classifier3(x)

        return x


from torchsummary import summary

model = VGG()

torch.no_grad()
reset()
rt = PowerSampler(0.02, sampler, "World")
array = []
for depth in range(0, 2):
    # 1, 256, 6, 6

    random_data = torch.rand((1, 3, 455, 455))
    with torch.no_grad():
        rt.start()
        now = time.time()
        result = model(random_data)
        duration = time.time() - now
        rt.stop()

        mutex.acquire()
        array.append(
            (duration, board_power / float(count), cpu_power / float(count), ram_power / float(count)))
        reset()
        mutex.release()
print("duration,board,cpu,ram")
for i in range(len(array)):
    print("{},{},{},{}".format(array[i][0], array[i][1], array[i][2], array[i][3]))
