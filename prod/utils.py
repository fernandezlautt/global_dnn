import torch
from torchvision import transforms
import torchvision

SIZE_IMAGES = (224, 224)

transformations = transforms.Compose(
    [
        transforms.Resize(SIZE_IMAGES),
        # transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


import torch
import torchvision
from unet_18 import ResNetUNet
from unet_50 import ResNetUNet50

resnet50 = torchvision.models.resnet50(pretrained=True)
resnet18 = torchvision.models.resnet18(pretrained=True)

# Init models
unet18 = ResNetUNet(resnet18)

unet50_basic = ResNetUNet50(resnet50)

unet50 = ResNetUNet50(resnet50)


unet18.load_state_dict(
    torch.load(
        "/mount/src/global_dnn/prod/model_25_normalized.pt",
        weights_only=True,
        map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
    )
)

unet50_basic.load_state_dict(
    torch.load(
        "/mount/src/global_dnn/prod/model_50_res_35.pt",
        weights_only=True,
        map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
    )
)

unet50.load_state_dict(
    torch.load(
        "/mount/src/global_dnn/prod/model_50_res_60.pt",
        weights_only=True,
        map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
    )
)


def get_image(image):
    # preprocess image
    img = transforms.ToTensor()(image)
    img = img.to("cuda" if torch.cuda.is_available() else "cpu")
    img = img[:3]
    img = transforms.Resize(SIZE_IMAGES)(img)
    return transforms.ToPILImage()(img)


def transform_to_night(image, model_name):
    """
    Transforms the input image into a nighttime scene using the selected model.
    """
    # model
    model = (
        unet18
        if model_name == "unet18"
        else unet50_basic if model_name == "unet50_basic" else unet50
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # preprocess image
    img = transforms.ToTensor()(image)
    img = img.to("cuda" if torch.cuda.is_available() else "cpu")
    img = img[:3]
    img = transformations(img)
    img = img.unsqueeze(0)

    # inference
    with torch.no_grad():
        out: torch.Tensor = model(img).cpu()

    out = out.squeeze(0)

    out = out.clamp(0, 1)

    out_img = torchvision.transforms.ToPILImage()(out)
    return out_img
