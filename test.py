import torch
import torch.nn as nn
from codecarbon import OfflineEmissionsTracker
import tensorflow

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.l = 1
        self.beg = True
        self.feature1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(4, 4), padding=2),
            nn.ReLU(inplace=True)
        )
        self.feature11 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.feature2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=(3, 3), padding=2),
            nn.ReLU(inplace=True))
        self.feature22 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.feature3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        self.feature4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        self.feature5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        self.feature55 = nn.Sequential(

            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.classifier1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True)
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

        self.layers = [
            self.feature1,
            self.feature11,
            self.feature2,
            self.feature22,
            self.feature3,
            self.feature4,
            self.feature5,
            self.feature55,
            self.classifier1,
            self.classifier2,
            self.classifier3
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers)):
            x = self.layers[i]
        return x


model = AlexNet()
torch.no_grad()

for depth in range(0, 500):
    if depth % 100 == 0:
        print("==========DEPTH {} ==========".format(depth))
    tracker = OfflineEmissionsTracker(country_iso_code="ITA",
                                      measure_power_secs=0.001,
                                      experiment_id=depth,
                                      log_level="critical")
    random_data = torch.rand((1, 3, 255, 255))

    with torch.no_grad():
        tracker.start()
        result = model(random_data)
        emission = tracker.stop()
    # print("====================>data {}".format(data))
