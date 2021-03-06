import torch.nn as nn
import torchvision

from libs.models.net import EasyNet, BenchmarkNet

__all__ = ["get_model"]

model_names = ["resnet18", "resnet34", "easynet", "benchmarknet"]


"""
Copyright (c) 2020 yiskw713
"""


def get_model(name: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name not in model_names:
        raise ValueError(
            """There is no model appropriate to your choice.
            You have to choose vgg11, resnet18, resnet34 as a model.
            """
        )

    if name == "easynet":
        model = EasyNet()
        return model
    
    if name == "benchmarknet":
        model = BenchmarkNet()
        return model

    print("{} will be used as a model.".format(name))
    model = getattr(torchvision.models, name)(pretrained=pretrained)
    if name.startswith("resnet"):
        model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=n_classes, bias=True)
    elif name.startswith("vgg"):
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=in_features, out_features=n_classes, bias=True)
    elif name.startswith("inception"):
        model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=n_classes, bias=True)
    

    return model
