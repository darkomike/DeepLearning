# DeepLearning

This is a Python project for deep learning experiments using PyTorch. It includes implementations of various neural network architectures (fully connected, convolutional, recurrent, GRU, and LSTM) trained on the MNIST dataset for handwritten digit recognition, as well as advanced models like sequence-to-sequence for machine translation and pre-training/fine-tuning on CIFAR-10.

## Files

- `simple_neural_network.py`: A simple fully connected neural network implementation with detailed comments explaining each line of code.
- `simple_convolution_neural_network.py`: A convolutional neural network implementation with detailed comments explaining each line of code.
- `simple_recurrent_neural_network.py`: A recurrent neural network (RNN) implementation with detailed comments explaining each line of code.
- `simple_gated_recurrent_unit_neural_network.py`: A gated recurrent unit (GRU) neural network implementation with detailed comments explaining each line of code.
- `simple_long_short_term_memory_neural_network.py`: A long short-term memory (LSTM) neural network implementation with detailed comments explaining each line of code.
- `simple_bidirectional_lstm_neural_network.py`: A bidirectional long short-term memory (BiLSTM) neural network implementation with detailed comments explaining each line of code.
- `pytorch_loadsave.py`: A convolutional neural network implementation demonstrating model checkpoint saving and loading functionality with detailed comments explaining each line of code.
- `sequence_to_sequence_model.py`: A sequence-to-sequence model for German to English translation using LSTM encoder-decoder architecture with detailed comments.
- `pretrain_finetune.py`: Implementation of pre-training and fine-tuning a VGG16 model on CIFAR-10 dataset with detailed comments.
- `utils.py`: Utility functions for saving and loading model checkpoints.
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

## Sequence-to-Sequence Model

This implementation demonstrates a neural machine translation system using an encoder-decoder architecture with LSTM layers for German to English translation.

### Architecture
- **Encoder**: LSTM-based encoder that processes the source German sentence and produces context vectors (hidden and cell states).
- **Decoder**: LSTM-based decoder that generates the target English sentence token by token, using the encoder's context.
- **Sequence-to-Sequence**: Combines encoder and decoder with teacher forcing during training.

### Key Components
- Tokenization using spaCy for German and English
- Vocabulary building with maximum size 1000 and minimum frequency 2
- Embedding layers for both encoder and decoder
- Dropout for regularization
- Teacher forcing ratio of 0.5 during training

Trained using Adam optimizer with weight decay and Cross-Entropy loss for 20 epochs with batch size 64. Uses TensorBoard for loss visualization and checkpoint saving.

## Pre-training and Fine-tuning

This implementation shows how to leverage pre-trained models for transfer learning by fine-tuning a VGG16 model (pre-trained on ImageNet) for CIFAR-10 classification.

### Approach
- Load pre-trained VGG16 model from torchvision
- Freeze all pre-trained parameters to retain learned features
- Replace the classifier head with a custom fully connected network (512 → 100 → 10)
- Fine-tune only the new classifier layers on CIFAR-10

### Key Features
- Uses Identity layer to bypass avgpool for custom classifier
- CIFAR-10 dataset with automatic download
- Batch size 1024 for efficient training
- Adam optimizer with learning rate 0.001

Trained for 5 epochs, demonstrating transfer learning benefits for image classification tasks.

## Requirements

- Python 3.x
- PyTorch (CPU or GPU version)
- torchvision (for MNIST and CIFAR-10 datasets)
- torchtext (for sequence-to-sequence model)
- spaCy with German and English models (for tokenization)

Install dependencies:
```bash
pip install torch torchvision torchtext spacy
python -m spacy download de
python -m spacy download en
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
python sequence_to_sequence_model.py
python pretrain_finetune.py
```

Each script will:
- Download the required datasets automatically (MNIST, Multi30k, CIFAR-10)
- Train the respective model
- Evaluate accuracy on training and test sets (where applicable)

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

### Sequence-to-Sequence Model
After running `sequence_to_sequence_model.py`, you'll see training progress and checkpoint saving:
```
=> Saving checkpoint...
For epoch 0 / 20
For epoch 1 / 20
...
```
The model trains for German to English translation using the Multi30k dataset. Loss is logged to TensorBoard in the `runs/loss_plot` directory.

### Pre-training and Fine-tuning
After running `pretrain_finetune.py`, you'll see accuracy results on CIFAR-10:
```
Checking accuracy on training data
Got X/50000 with accuracy XX.XX%
Checking accuracy on test data
Got Y/10000 with accuracy YY.YY%
```
The pre-trained VGG16 model is fine-tuned for CIFAR-10 classification, demonstrating transfer learning.

## Contributing

Feel free to modify the network architecture, hyperparameters, or add more features!