import time

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import threading as th

from keras_flops import get_flops

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
            self._timer = th.Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


def sampler(board="TX2"):
    global count, ram_power, cpu_power, board_power, gpu_power, mutex
    mutex.acquire()
    count += 1
    if board == "TX2":
        with open("/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power2_input", 'r') as f:
            ram_power += float(f.read())
        with open("/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power0_input", 'r') as f:
            board_power += float(f.read())
        with open("/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power1_input", 'r') as f:
            cpu_power += float(f.read())
        with open("/sys/bus/i2c/drivers/ina3221x/0-0040/iio_device/in_power0_input", 'r') as f:
            gpu_power += float(f.read())
    elif board == "XAVIER":
        with open("/sys/bus/i2c/drivers/ina3221x/7-0040/iio:device0/in_power1_input", 'r') as f:
            cpu_power += float(f.read())
        with open("/sys/bus/i2c/drivers/ina3221x/7-0040/iio:device0/in_power0_input", 'r') as f:
            board_power += float(f.read())
    mutex.release()


def reset():
    global count, ram_power, cpu_power, board_power, gpu_power, mutex
    count = ram_power = cpu_power = board_power = gpu_power = 0


class AlexNet(Sequential):
    def __init__(self, input_shape=(3, 455, 455), num_classes=1000):
        super().__init__()

        self.add(Conv2D(96, kernel_size=(11, 11), strides=4,
                        padding='valid', activation='relu',
                        input_shape=input_shape,
                        kernel_initializer='he_normal'))

        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='valid', data_format=None
                              #, input_shape=input_shape
                              ))

        self.add(Conv2D(256, kernel_size=(5, 5), strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal'
                        #, input_shape=input_shape
                        ))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='valid', data_format=None
                              #, input_shape=input_shape
                              ))

        self.add(Conv2D(384, kernel_size=(3, 3), strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal'
                        #, input_shape=input_shape
                        ))

        self.add(Conv2D(384, kernel_size=(3, 3), strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal'
                        , input_shape=input_shape
                        ))

        self.add(Conv2D(256, kernel_size=(3, 3), strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal'
                        , input_shape=input_shape
                        ))

        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='valid', data_format=None
                              , input_shape=input_shape
                              ))

        self.add(Flatten()
                 #, input_shape=input_shape
                 )
        self.add(Dense(4096, activation='relu'
                       #, input_shape=input_shape
                       ))
        self.add(Dense(4096, activation='relu'
                       #, input_shape=input_shape
                       ))
        self.add(Dense(1000, activation='relu'
                       #, input_shape=input_shape
                       ))
        self.add(Dense(num_classes, activation='softmax'
                       #, input_shape=input_shape
                       ))

model = AlexNet((455, 455, 3), 10000)
print(isinstance(model, tf.keras.Sequential))
flops = get_flops(model)
print(f"FLOPS: {flops / 10 ** 9:.03} G")

model = AlexNet((112, 112, 96), 1000)
print(model.summary())
#211,848,952
#      34944
#211,814,008
exit(0)
reset()
rt = PowerSampler(0.02, sampler, "XAVIER")
array = []
for depth in range(0, 2):
    # 1, 256, 6, 6

    a = tf.random.normal([1, 455, 455, 3], 0, 255, tf.float32, seed=1)

    rt.start()
    now = time.time()
    result = model.predict(a)
    duration = time.time() - now
    rt.stop()

    mutex.acquire()
    array.append(
        (duration, board_power / float(count), cpu_power / float(count)))
    reset()
    mutex.release()

print("duration,board,cpu,ram")
for i in range(len(array)):
    print("{},{},{}".format(array[i][0], array[i][1], array[i][2]))
