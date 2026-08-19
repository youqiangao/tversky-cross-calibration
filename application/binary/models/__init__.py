from .fcn8 import FCN8
from .unet import UNet


def build_model(name: str, num_classes: int = 2):
    model_name = name.lower()
    if model_name == "unet":
        return UNet(num_classes=num_classes)
    if model_name == "fcn8":
        return FCN8(num_classes=num_classes)
    raise ValueError(f"Unsupported model '{name}'. Expected 'unet' or 'fcn8'.")
