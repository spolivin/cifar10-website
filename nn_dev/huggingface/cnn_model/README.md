---
license: mit
pipeline_tag: image-classification
---
# ResNet-20 model for CIFAR-10

This is the repo for a custom-made neural network based on ResNet architecture that has been trained on CIFAR-10 dataset.

## Model details

- **Architecture:** ResNet
- **Input shape:** 3x32x32
- **Output classes:** 10
- **Parameters:** 272,474
- **Dataset:** CIFAR-10

More details can be found in the `config.json` file inside this repository and the original [git repository](https://github.com/spolivin/cifar10-website/blob/master/nn_dev/pytorch_models/architectures.py) from which the model originated.

## Model usage

Since the model for which the weights loaded in this repository are intended is a part of a custom Python package, one needs to firstly clone the project locally:

```bash
git clone https://github.com/spolivin/cifar10-website.git
cd cifar10-website/nn_dev
```

Next, in a Python script we can make imports and load the weights:

```python
import torch

from pytorch_models import resnet20

# URL from which to load the weights
URL = "https://huggingface.co/spolivin/cnn-cifar10/resolve/main/resnet20_weights.pth"

# Building the ResNet-20 model
resnet20_model = resnet20()

# Loading the pretrained weights to the model
resnet20_model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            url=URL,
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )
```
