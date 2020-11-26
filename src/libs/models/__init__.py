import torch.nn as nn
import torchvision

__all__ = ["get_model"]

model_names = ["resnet18", "resnet34", "resnet50", "vgg16", "inception_v3"]


def get_model(name: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name not in model_names:
        raise ValueError(
            """There is no model appropriate to your choice.
            You have to choose resnet18, resnet34, resnet50 as a model.
            """
        )

    print("{} will be used as a model.".format(name))
    model = getattr(torchvision.models, name)(pretrained=pretrained)
    if name.startswith("resnet"):
        model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=n_classes, bias=True)
    elif name.startswith("vgg"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=in_features, out_features=n_classes, bias=True)
    elif name.startswith("inception"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=n_classes, bias=True)
    

    return model
