
# FNIST Pytorch CNN No Regularization then Regularization
![image](https://github.com/FarzadMalik/FNIST_pytorch_CNN_no_regularization_then_regularization/assets/107833662/7c5f458d-3a15-404f-a817-00a2ffa644ec)



This repository contains code for training a Convolutional Neural Network (CNN) to classify the Fashion MNIST dataset using PyTorch. The model is trained first without regularization and then with regularization techniques such as Batch Normalization and Dropout to improve generalization and reduce overfitting.



## Dataset
![image](https://github.com/FarzadMalik/FNIST_pytorch_CNN_no_regularization_then_regularization/assets/107833662/f4a9e669-4578-4459-85bd-d7e0cbab6a9f)

The Fashion MNIST dataset consists of 28x28 grayscale images of fashion items. It is a popular dataset used for image classification tasks.

## Requirements

- Python (>=3.6)
- PyTorch (>=1.7)
- torchvision (>=0.8)

Install the required packages using pip:

```bash
pip install torch torchvision
```

## Data Augmentation and Regularization

Data augmentation is performed on the training set using various transformations such as random cropping, flipping, rotation, color jitter, perspective transform, random erasing, and Gaussian blur. Regularization techniques like Batch Normalization and Dropout are implemented in the neural network to improve model performance and reduce overfitting.

## Neural Network Architecture

The neural network architecture used for this classification task is as follows:

1. Convolutional Layer 1 (input channels: 1, output channels: 32, kernel size: 3x3)
2. Batch Normalization Layer 1
3. Dropout Layer 1 (dropout rate: 0.2)
4. ReLU Activation
5. Max Pooling Layer (kernel size: 2x2, stride: 2)
6. Convolutional Layer 2 (input channels: 32, output channels: 64, kernel size: 3x3)
7. Batch Normalization Layer 2
8. Dropout Layer 2 (dropout rate: 0.2)
9. ReLU Activation
10. Max Pooling Layer (kernel size: 2x2, stride: 2)
11. Fully Connected Layer 1 (input size: 64x12x12, output size: 128)
12. ReLU Activation
13. Fully Connected Layer 2 (input size: 128, output size: 10, representing the 10 classes)
14. No Activation (Logits)

![image](https://github.com/FarzadMalik/FNIST_pytorch_CNN_no_regularization_then_regularization/assets/107833662/9e993730-04fd-47a5-92c5-c0aa0a3b7398)


## Training

The model is trained using Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.001, momentum of 0.9, and L2 weight decay of 0.001.

Training is performed for 25 epochs with a batch size of 32.

## Evaluation

After each epoch, the model is evaluated on the test dataset to monitor its performance.

## Results

The training progress, including the epoch number, accuracy, and actual loss, is displayed during the training process.

## File Structure

The repository includes the following files:

- `fashion_mnist_pytorch_cnn.pth`: The trained model weights saved as a PyTorch model checkpoint file.
- `training_pytorch_cnn.ipynb`: The Jupyter Notebook containing the code for training the model with regularization.
- `.gitattributes` and `.ipynb_checkpoints`: Git-related files automatically generated during repository initialization.

## Usage

To train the model, open and run the `training_pytorch_cnn.ipynb` notebook in your Jupyter environment.

## License

This project is licensed under the [MIT License](LICENSE).
```

