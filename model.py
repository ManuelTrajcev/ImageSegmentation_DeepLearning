import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second conv
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # pooling layer

        # UNET Down part

        for feature in features:  # Map 1 to 64
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # UNET Up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2  # 512*2, double height and width
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))  # 2 cols right, one row up

        # Bottom layer - bottleneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)  # 512, last one

        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse the list

        for index in range(0, len(self.ups), 2):  # up and double conv
            x = self.ups[index](x)
            skip_connection = skip_connections[index // 2]

            if x.shape != skip_connection.shape:        #za da raboti i so broevi neddelivi so 2
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[index+1](concat_skip)

        return self.final_conv(x)

def test():
    x  = torch.randn((3,1,161,161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()