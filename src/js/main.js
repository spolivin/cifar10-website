// Importing SCSS styles and modules
import '../scss/styles.scss'
import * as tf from "@tensorflow/tfjs"
import {InferenceSession, Tensor} from "onnxjs"

// Setting document constants
const imageInput = document.getElementById("formFile");
const displayImage = document.getElementById("displayImage");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const divelem = document.getElementById("predicted-class");
const response = document.getElementById("response");

// Mean and standard deviation values for normalizing image
const means = [0.4914, 0.4822, 0.4465];
const stds = [0.247,  0.243,  0.261];

// Classes to predict for an image
const classes = [
        'plane', 'car', 'bird',
        'cat', 'deer', 'dog',
        'frog', 'horse', 'ship', 'truck'
];
const model_path = './assets/onnx_model.onnx';

// Setting event listener triggered upon image upload
imageInput.addEventListener("change", function() {
    divelem.style.display = "none";
    const file = imageInput.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = function(e) {
        displayImage.src = e.target.result;
        displayImage.style.display = "block";
        // After image has been uploaded, drawing it onto the canvas
        displayImage.onload = function() {
          canvas.width = displayImage.width;
          canvas.height = displayImage.height;
          ctx.drawImage(displayImage, 0, 0);
          const tensor = preprocessImage(displayImage);
          makeInference(tensor);
        };
      };
      reader.readAsDataURL(file);
    }
  });

// Function for resizing, normalizing and preparing image for the NN
function preprocessImage(imageData) {
    // Converting image data to tensor
    let tensor = tf.browser.fromPixels(imageData);
    // Resizing the image tensor to 32x32
    tensor = tf.image.resizeBilinear(tensor, [32, 32]);
    tensor = tensor.slice([0, 0, 0], [32, 32, 3]);
    // Normalizing the image
    tensor = tensor.div(255.0);
    tensor = tensor.sub(means).div(stds);
    // Rearranging the tensor dimensions
    tensor = tensor.transpose([2, 0, 1]); // Change from [32, 32, 3] to [3, 32, 32]
    // Adding batch dimenstion
    tensor = tensor.expandDims(0);
    tensor = tensor.dataSync();
    const preprocessedTensor = new Tensor(tensor, "float32", [1, 3, 32, 32]);
    return preprocessedTensor
  }

// Function for predicting the image class and displaying the result
async function makeInference(input) {
    const wrapper = document.createElement("div");
    wrapper.innerHTML = [
      '<div class="spinner-border" role="status">',
      '<span class="visually-hidden">Loading...</span>',
      '</div>'
    ].join('');
    response.append(wrapper);
    // Running inference
    const sess = new InferenceSession();
    await sess.loadModel(model_path);
    const outputMap = await sess.run([input]);
    wrapper.style.display = "none";
    // Making a prediction
    const outputTensor = outputMap.values().next().value;
    const predictedClass = classes[indexOfMax(outputTensor.data)];
    divelem.style.display = "block";
    divelem.innerHTML = `Predicted class: ${predictedClass}`;
  }

// Function for finding the largest number in an array
function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
  }
