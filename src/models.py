"""
Submodule for simple model defining.
"""



import torch
import math
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
import segmentation_models_pytorch as smp



class Up(nn.Module):
    """
    Single upscale element for UNet model.
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet34(nn.Module):
    """
    UNet model with simplified ResNet34 encoder averaging.
    """
    def __init__(self, in_channels=7, num_classes=4, pretrained=True):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        net = resnet34(weights=weights)

        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                mean_w = net.conv1.weight.mean(dim=1, keepdim=True)   # (64,1,7,7)
                self.conv1.weight.copy_(mean_w.repeat(1, in_channels, 1, 1))

        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        # encoder stages (ResNet34)
        self.layer1 = net.layer1   # 64
        self.layer2 = net.layer2   # 128
        self.layer3 = net.layer3   # 256
        self.layer4 = net.layer4   # 512

        # decoder stages
        self.up4 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up1 = Up(64, 64, 64)
        self.up0 = Up(64, 0, 32)
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))   # 1/2
        x1 = self.layer1(self.maxpool(x0))        # 1/4
        x2 = self.layer2(x1)                      # 1/8
        x3 = self.layer3(x2)                      # 1/16
        x4 = self.layer4(x3)                      # 1/32

        d = self.up4(x4, x3)
        d = self.up3(d, x2)
        d = self.up2(d, x1)
        d = self.up1(d, x0)
        d = self.up0(d)
        return self.head(d)                       # (B, 4, H, W)


class UNetPlusPlusResNet34(smp.UnetPlusPlus):
    """
    SMP UNet++ wrapper.
    """
    def __init__(self, in_channels=7, num_classes=4, pretrained=None):
        super().__init__(
            encoder_name="resnet34",
            encoder_weights=pretrained,
            in_channels=7,
            classes=4,
        )


class CosineAnnealingWarmRestartsDecay(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    """
    Custom CosineAnnealingWarmRestart scheduler, used to decrease max and min lr for each repeating.
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, decay=1):
        super().__init__(optimizer, T_0, T_mult=T_mult,
                         eta_min=eta_min, last_epoch=last_epoch)
        self.decay = decay
        self.initial_lrs = self.base_lrs[:]
        self.initial_eta_min = eta_min

    def step(self, epoch=None):
        if epoch is None:
            if self.T_cur + 1 == self.T_i:
                self.base_lrs = [lr * self.decay for lr in self.base_lrs]
                self.eta_min  = self.eta_min * self.decay
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            else:
                n = 0
            self.base_lrs = [lr * (self.decay ** n) for lr in self.initial_lrs]
            self.eta_min  = self.initial_eta_min * (self.decay ** n)
        super().step(epoch)