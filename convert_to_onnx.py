import onnx
import torch

from nn_dev.pytorch_models import ResNet, resnet20

# URL for pretrained model weights on HuggingFace
WEIGHTS_URL = "https://huggingface.co/spolivin/cnn-cifar10/resolve/main/resnet20_weights.pth"
# Path for saving the pretrained model in ONNX format
ONNX_MODEL_PATH = "src/assets/onnx_model.onnx"
# Example input
MODEL_EXAMPLE_INPUT = torch.rand(1, 3, 32, 32)
# Device to be used when loading weights from HuggingFace
DEVICE = torch.device("cpu")


def load_pretrained_weights(url: str) -> ResNet:
    """Loads pretrained weights from HuggingFace repository.

    Args:
        url (str): Path to model weights on HuggingFace.

    Returns:
        ResNet: Instance of ResNet-20 model with pretrained weights.
    """
    # Creating an empty ResNet-20 model
    net = resnet20()
    print("Loading model weights from Hugging Face...")
    # Loading pretrained weights from HF and loading them into the model
    net.load_state_dict(
        torch.hub.load_state_dict_from_url(
            url=url, weights_only=True, map_location=DEVICE
        )
    )
    print("Weights loaded")

    return net


if __name__ == "__main__":
    # Getting the model with pretrained weights
    net = load_pretrained_weights(url=WEIGHTS_URL)
    print("Converting pretrained model to ONNX format...")
    # Exporting model to ONNX format
    torch.onnx.export(
        model=net,
        args=MODEL_EXAMPLE_INPUT,
        f=ONNX_MODEL_PATH,
        verbose=False,
        opset_version=9,
    )
    print("Model successfully converted to ONNX")
    print("Validating ONNX model...")
    # Validating the saved ONNX model
    saved_onnx_model = onnx.load(ONNX_MODEL_PATH)
    onnx.checker.check_model(model=saved_onnx_model)
    print("Model has been successfully validated")
