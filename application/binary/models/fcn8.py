from __future__ import annotations

import inspect

import torch
import torch.nn as nn
from torchvision import models


def _get_vgg16_backbone(pretrained: bool = False):
    signature = inspect.signature(models.vgg16)
    if "weights" in signature.parameters:
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        return models.vgg16(weights=weights)
    return models.vgg16(pretrained=pretrained)


def _upsampling_kernel(in_channels: int, out_channels: int, kernel_size: int) -> torch.Tensor:
    factor = (kernel_size + 1) // 2
    center = factor - 1 if kernel_size % 2 == 1 else factor - 0.5
    og = torch.arange(kernel_size, dtype=torch.float32)
    filt = (1 - torch.abs(og - center) / factor).unsqueeze(0)
    kernel = filt.t() * filt
    weight = torch.zeros(in_channels, out_channels, kernel_size, kernel_size)
    for channel in range(min(in_channels, out_channels)):
        weight[channel, channel] = kernel
    return weight


class FCN8(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained_backbone: bool = False) -> None:
        super().__init__()
        vgg = _get_vgg16_backbone(pretrained=pretrained_backbone)
        features = list(vgg.features.children())
        classifier = list(vgg.classifier.children())

        features[0].padding = (100, 100)
        for layer in features:
            if isinstance(layer, nn.MaxPool2d):
                layer.ceil_mode = True

        self.pool3 = nn.Sequential(*features[:17])
        self.pool4 = nn.Sequential(*features[17:24])
        self.pool5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.conv6 = nn.Conv2d(512, 4096, kernel_size=7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        if pretrained_backbone:
            self.conv6.weight.data.copy_(classifier[0].weight.data.view(self.conv6.weight.shape))
            self.conv6.bias.data.copy_(classifier[0].bias.data)
            self.conv7.weight.data.copy_(classifier[3].weight.data.view(self.conv7.weight.shape))
            self.conv7.bias.data.copy_(classifier[3].bias.data)

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        self.upscore2.weight.data.copy_(_upsampling_kernel(num_classes, num_classes, 4))
        self.upscore_pool4.weight.data.copy_(_upsampling_kernel(num_classes, num_classes, 4))
        self.upscore8.weight.data.copy_(_upsampling_kernel(num_classes, num_classes, 16))

        for module in (self.upscore2, self.upscore_pool4, self.upscore8):
            module.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_h, input_w = x.shape[-2:]
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)
        score = self.score_fr(self.drop7(self.relu7(self.conv7(self.drop6(self.relu6(self.conv6(pool5)))))))
        upscore2 = self.upscore2(score)

        score_pool4 = self.score_pool4(0.01 * pool4)
        score_pool4 = score_pool4[:, :, 5 : 5 + upscore2.size(2), 5 : 5 + upscore2.size(3)]
        fuse_pool4 = self.upscore_pool4(score_pool4 + upscore2)

        score_pool3 = self.score_pool3(0.0001 * pool3)
        score_pool3 = score_pool3[:, :, 9 : 9 + fuse_pool4.size(2), 9 : 9 + fuse_pool4.size(3)]
        out = self.upscore8(score_pool3 + fuse_pool4)
        return out[:, :, 31 : 31 + input_h, 31 : 31 + input_w].contiguous()
