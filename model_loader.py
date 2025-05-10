from torchvision import models
import torch.nn as nn

def load_model(model_type="resnet18", pretrained=True, num_classes=101):
    if model_type == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "alexnet":
        model = models.alexnet(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError("Unsupported model type.")
    return model
