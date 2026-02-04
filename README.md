# DeepLearning

This is a Python project for deep learning experiments using PyTorch. It includes implementations of both a simple fully connected neural network and a convolutional neural network, trained on the MNIST dataset for handwritten digit recognition.

## Files

- `simple_neural_network.py`: A simple fully connected neural network implementation with detailed comments explaining each line of code.
- `simple_convolution_neural_network.py`: A convolutional neural network implementation with detailed comments explaining each line of code.
- `.gitignore`: Ignores the dataset folder to avoid committing large data files.
- `README.md`: This file, providing project overview and instructions.

## Neural Network Architectures

### Fully Connected Neural Network
The simple neural network consists of:
- Input layer: 784 neurons (28x28 flattened MNIST images)
- Hidden layer: 50 neurons with ReLU activation
- Output layer: 10 neurons (one for each digit 0-9)

Trained using Adam optimizer and Cross-Entropy loss for 1 epoch with batch size 64.

### Convolutional Neural Network
The CNN consists of:
- Conv1: 1 input channel → 8 output channels, 3x3 kernel, stride 1, padding 1
- MaxPool: 2x2 kernel, stride 2
- Conv2: 8 input channels → 16 output channels, 3x3 kernel, stride 1, padding 1
- MaxPool: 2x2 kernel, stride 2
- Fully Connected: 16*7*7 → 10 neurons

Trained using Adam optimizer and Cross-Entropy loss for 5 epochs with batch size 64.

## Requirements

- Python 3.x
- PyTorch (CPU or GPU version)
- torchvision (for MNIST dataset)

Install dependencies:
```bash
pip install torch torchvision
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/darkomike/DeepLearning.git
cd DeepLearning
```

2. Run the fully connected neural network:
```bash
python simple_neural_network.py
```

3. Run the convolutional neural network:
```bash
python simple_convolution_neural_network.py
```

Both scripts will:
- Download the MNIST dataset automatically
- Train the respective model
- Evaluate accuracy on training and test sets

## Dataset

The MNIST dataset is downloaded automatically by torchvision. If you prefer to use a local copy, place it in the `dataset/` folder (this folder is ignored by git to avoid committing large files).

## Output

### Fully Connected Neural Network
After running `simple_neural_network.py`, you'll see accuracy results like:
```
Checking accuracy on training data
Got 58974/60000 with accuracy 98.29%
Checking accuracy on test data
Got 9654/10000 with accuracy 96.54%
```

### Convolutional Neural Network
After running `simple_convolution_neural_network.py`, you'll see accuracy results like:
```
Checking accuracy on training data
Got 59750/60000 with accuracy 99.58%
Checking accuracy on test data
Got 9875/10000 with accuracy 98.75%
```

## Contributing

Feel free to modify the network architecture, hyperparameters, or add more features!