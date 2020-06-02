
from torchvision import models

def ResnetBackbone(layer_num=18, pretrained=True):
    """
    Use predefined resnet for encoding (generally 18 or 50)
    """
    if layer_num == 18:
        model = models.resnet18(pretrained=pretrained, progress=False)
    elif layer_num == 50:
        model = models.resnet50(pretrained=pretrained, progress=False)
    else:
        RuntimeError("Choose appropriate resnet model")

    return model

