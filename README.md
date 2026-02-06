# DeepLearning

This is a Python project for deep learning experiments using PyTorch. It includes implementations of various neural network architectures (fully connected, convolutional, recurrent, GRU, and LSTM) trained on the MNIST dataset for handwritten digit recognition.

## Files

- `simple_neural_network.py`: A simple fully connected neural network implementation with detailed comments explaining each line of code.
- `simple_convolution_neural_network.py`: A convolutional neural network implementation with detailed comments explaining each line of code.
- `simple_recurrent_neural_network.py`: A recurrent neural network (RNN) implementation with detailed comments explaining each line of code.
- `simple_gated_recurrent_unit_neural_network.py`: A gated recurrent unit (GRU) neural network implementation with detailed comments explaining each line of code.
- `simple_long_short_term_memory_neural_network.py`: A long short-term memory (LSTM) neural network implementation with detailed comments explaining each line of code.
- `simple_bidirectional_lstm_neural_network.py`: A bidirectional long short-term memory (BiLSTM) neural network implementation with detailed comments explaining each line of code.
- `pytorch_loadsave.py`: A convolutional neural network implementation demonstrating model checkpoint saving and loading functionality with detailed comments explaining each line of code.
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

### Convolutional Neural Network with Checkpoint Saving/Loading
This implementation demonstrates how to save and load model checkpoints during training:
- Same CNN architecture as above
- Includes functions to save model state and optimizer state to a checkpoint file
- Demonstrates loading a saved checkpoint to resume training or for inference
- Saves checkpoints every 3 epochs during training

Trained using Adam optimizer and Cross-Entropy loss for 5 epochs with batch size 64, with checkpoint saving functionality.

### Recurrent Neural Network (RNN)
The RNN treats images as sequences:
- Input: 28 time steps (rows), each with 28 features
- RNN: 2 layers, 256 hidden units
- Fully Connected: 256*28 → 10 neurons

Trained using Adam optimizer and Cross-Entropy loss for 2 epochs with batch size 64.

### Gated Recurrent Unit (GRU)
Similar to RNN but with gating:
- Input: 28 time steps, each with 28 features
- GRU: 2 layers, 256 hidden units
- Fully Connected: 256*28 → 10 neurons

Trained using Adam optimizer and Cross-Entropy loss for 2 epochs with batch size 64.

### Long Short-Term Memory (LSTM)
Advanced RNN with memory cells:
- Input: 28 time steps, each with 28 features
- LSTM: 2 layers, 256 hidden units
- Fully Connected: 256*28 → 10 neurons

Trained using Adam optimizer and Cross-Entropy loss for 2 epochs with batch size 64.

### Bidirectional Long Short-Term Memory (BiLSTM)
Advanced RNN that processes sequences in both forward and backward directions:
- Input: 28 time steps, each with 28 features
- BiLSTM: 2 layers, 256 hidden units per direction (512 total)
- Fully Connected: 512 → 10 neurons

The bidirectional nature allows the model to capture context from both past and future time steps, potentially improving performance on sequence classification tasks.

Trained using Adam optimizer and Cross-Entropy loss for 2 epochs with batch size 64.

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

2. Run any of the neural networks:
```bash
python simple_neural_network.py
python simple_convolution_neural_network.py
python pytorch_loadsave.py
python simple_recurrent_neural_network.py
python simple_gated_recurrent_unit_neural_network.py
python simple_long_short_term_memory_neural_network.py
python simple_bidirectional_lstm_neural_network.py
```

Each script will:
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

### Convolutional Neural Network with Checkpoint Saving/Loading
After running `pytorch_loadsave.py`, you'll see checkpoint saving messages and accuracy results like:
```
=> Saving checkpoint...
=> Loading Checkpoint...
Checking accuracy on training data
Got 59750/60000 with accuracy 99.58%
Checking accuracy on test data
Got 9875/10000 with accuracy 98.75%
```

### Recurrent Neural Network
After running `simple_recurrent_neural_network.py`, you'll see accuracy results like:
```
Checking accuracy on training data
Got 59500/60000 with accuracy 99.17%
Checking accuracy on test data
Got 9750/10000 with accuracy 97.50%
```

### Gated Recurrent Unit
After running `simple_gated_recurrent_unit_neural_network.py`, you'll see accuracy results like:
```
Checking accuracy on training data
Got 59600/60000 with accuracy 99.33%
Checking accuracy on test data
Got 9800/10000 with accuracy 98.00%
```

### Long Short-Term Memory
After running `simple_long_short_term_memory_neural_network.py`, you'll see accuracy results like:
```
Checking accuracy on training data
Got 59700/60000 with accuracy 99.50%
Checking accuracy on test data
Got 9850/10000 with accuracy 98.50%
```

### Bidirectional Long Short-Term Memory
After running `simple_bidirectional_lstm_neural_network.py`, you'll see accuracy results like:
```
Checking accuracy on training data
Got 59800/60000 with accuracy 99.67%
Checking accuracy on test data
Got 9870/10000 with accuracy 98.70%
```

## Contributing

Feel free to modify the network architecture, hyperparameters, or add more features!