# Developing Image Classifier

This directory is dedicated to developing a neural network that would perform well on *CIFAR-10* dataset. We will then use it for online web-application in order to be able to classify images.

## Custom PyTorch Trainer

Although training and validating models in PyTorch is fairly flexible, one needs to firstly understand the intricacies of writing training and validation loops from scratch.

In order to automate this process, I decided to create a custom Trainer that would just need some parameters to be passed in to run the training and validation without the need to write loops from scratch. More details on the API and example usages can be consulted in [this notebook](./training_cnns.ipynb) which has been used for training models on *GPU*.

## HuggingFace

As one can see at the end of the notebook, the weights of the best-performing model are saved. These are not tracked by this repository but instead are loaded to [other repository](https://huggingface.co/spolivin/cnn-cifar10) on HuggingFace. This has been done for the purpose of avoiding training the model again but instead just load the weights and launch the application without any bottlenecks.

One can consult more details about the resulting model in [Readme](./huggingface/cnn_model/README.md) and [configuration file](./huggingface/cnn_model/config.json) containing more information about training/validating/testing process.
