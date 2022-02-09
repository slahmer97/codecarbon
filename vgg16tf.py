import time

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel('ERROR')

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
            self._timer = th.Timer(self.interval, self._run)
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



class VGG(Sequential):
    def __init__(self, input_shape=(3, 255, 255), num_classes=1000):
        super().__init__()

        self.add(
            Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')
        )

        self.add(
            Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')
        )

        self.add(
            MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')
        )

        self.add(
            Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')
        )
        self.add(
            Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')
        )
        self.add(
            MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')
        )
        self.add(
            Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')
        )
        self.add(
            Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')
        )
        self.add(
            Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')
        )

        self.add(
            MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')
        )

        self.add(
            Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')
        )
        self.add(
            Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')
        )
        self.add(
            Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')
        )
        self.add(
            MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')
        )
        self.add(
            Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')
        )
        self.add(
            Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')
        )
        self.add(
            Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')
        )
        self.add(
            MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')
        )
        self.add(
            Flatten(name='flatten')
        )
        self.add(
            Dense(4096, activation='relu', name='fc1')
        )
        self.add(
            Dense(4096, activation='relu', name='fc2')
        )
        self.add(
            Dense(num_classes, activation='softmax', name='predictions')
        )


model = VGG((455, 455, 3), 10000)
a = tf.random.normal([1, 455, 455, 3], 0, 255, tf.float32, seed=1)
result = model.predict(a, steps=1)

print(result.shape)
print(model.summary())
exit(0)
reset()
rt = PowerSampler(0.02, sampler, "World")
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
        (duration, board_power / float(count), cpu_power / float(count), ram_power / float(count)))
    reset()
    mutex.release()

print("duration,board,cpu,ram")
for i in range(len(array)):
    print("{},{},{},{}".format(array[i][0], array[i][1], array[i][2], array[i][3]))
