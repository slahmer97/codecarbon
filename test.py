import time
from threading import Timer
import torch
import torch.nn as nn
from codecarbon import OfflineEmissionsTracker

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
    global count, ram_power, cpu_power, board_power
    count += 1
    with open("/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power2_input", 'r') as f:
        ram_power += float(f.read())
    with open("/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power0_input", 'r') as f:
        board_power += float(f.read())
    with open("/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power1_input", 'r') as f:
        cpu_power += float(f.read())


def reset():
    global count, ram_power, cpu_power, board_power
    count = ram_power = cpu_power = board_power = 0


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=2),
            nn.ReLU(inplace=True)
        )
        self.features2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=(5,5), padding=2),
            nn.ReLU(inplace=True)
        )
        self.features4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.features5 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True)
        )
        self.features6 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True)
        )
        self.features7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
        )
        self.features8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features1(x)
        """
        x = self.features2(x)

        x = self.features3(x)

        x = self.features4(x)

        x = self.features5(x)

        x = self.features6(x)

        x = self.features7(x)


        x = self.features8(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier1(x)

        x = self.classifier2(x)
        x = self.classifier3(x)
        """
        return x

model = AlexNet()
torch.no_grad()
rt = PowerSampler(0.01, sampler, "World")
array = []
for depth in range(0, 5):
    reset()
    random_data = torch.rand((1, 3, 255, 255))
    with torch.no_grad():
        #rt.start()
        now = time.time()
        result = model(random_data)
        print(result.shape)
        exit(0)
        duration = time.time() - now
        #rt.stop()
        array.append(
            (duration, board_power / float(count), board_power / float(cpu_power), board_power / float(ram_power)))

print(array)
