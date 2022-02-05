import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from codecarbon import EmissionsTracker


class Net(nn.Module):
    def __init__(self, input_size=128, length=1, width=100, act=0):
        super(Net, self).__init__()
        self.activation_function = act
        self.hidden = []
        self.hidden_count = length
        self.width = width
        self.input = nn.Linear(input_size, self.width)
        for i in range(self.hidden_count):
            self.hidden.append(nn.Linear(self.width, self.width))
        self.output = nn.Linear(self.width, 1)

    # x represents our data
    def forward(self, x):

        act = F.relu

        x = self.input(x)
        for i in range(self.hidden_count):
            x = self.hidden[i](x)

        output = self.output(x)
        return output


width = 1000
length = 0
from codecarbon import OfflineEmissionsTracker

from torchvision import models

model = models.vgg16()
torch.no_grad()

for depth in range(0, 10):
    print("==========DEPTH {} ==========".format(depth))
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")
    random_data = torch.rand((1, 3, 255, 255))
    now = time.time()
    tracker.start()
    with torch.no_grad():
        #for _ in range(10):
        result = model(random_data)
    data = tracker.get_data()
    emission = tracker.stop()
    duration = time.time() - now
    print("====================>data {}".format(data))

