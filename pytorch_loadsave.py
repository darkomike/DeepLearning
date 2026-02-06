# PyTorch Model Checkpoint Saving and Loading Example
# This script demonstrates how to save and load PyTorch model checkpoints during training.
# It implements a Convolutional Neural Network (CNN) for MNIST digit classification,
# with functionality to save the model's state and optimizer state to a file,
# and load them back to resume training or perform inference.
# This is useful for long training sessions, resuming interrupted training,
# or deploying trained models.

# Import necessary libraries for PyTorch deep learning
import torch  # Main PyTorch library for tensor operations and neural networks
import torch.nn as nn  # Neural network modules (layers, loss functions, etc.)
import torch.optim as optim  # Optimization algorithms (Adam, SGD, etc.)
import torch.nn.functional as F  # Functional interface for neural network operations (relu, etc.)
from torch.utils.data import DataLoader  # Utilities for loading and batching datasets
import torchvision.datasets as datasets  # Pre-built datasets like MNIST
import torchvision.transforms as transforms  # Image transformations (ToTensor, etc.)

# Define a convolutional neural network class for image classification
# This CNN architecture is designed for the MNIST dataset (28x28 grayscale images)
# It consists of two convolutional layers followed by max pooling and a fully connected layer
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ConvolutionalNeuralNetwork, self).__init__()  # Initialize the parent nn.Module class
        # First convolutional layer: extracts 8 feature maps from 1 input channel (grayscale)
        # Kernel size 3x3, stride 1, padding 1 maintains spatial dimensions
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # Max pooling layer: reduces spatial dimensions by half (2x2 kernel, stride 2)
        # This helps reduce computational complexity and extract dominant features
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        # Second convolutional layer: extracts 16 feature maps from 8 input channels
        # Increases feature complexity while maintaining spatial size with padding
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # Fully connected layer: maps flattened convolutional features to class probabilities
        # Input size calculation: after two conv layers and two pooling operations:
        # 28x28 -> 28x28 -> 14x14 -> 14x14 -> 7x7, so 16 channels * 7 * 7 = 784 features
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        # Forward pass through the network
        # Apply first convolution followed by ReLU activation for non-linearity
        x = F.relu(self.conv1(x))
        # Apply max pooling to reduce spatial dimensions and extract dominant features
        x = self.pool(x)
        # Apply second convolution followed by ReLU activation
        x = F.relu(self.conv2(x))
        # Apply max pooling again
        x = self.pool(x)
        # Flatten the 3D tensor to 2D for the fully connected layer
        # x.shape[0] is batch size, -1 infers the flattened feature dimension
        x = x.reshape(x.shape[0], -1)
        # Apply fully connected layer (logits output, no activation - handled by loss function)
        x = self.fc1(x)
        return x

# Set the device to GPU if available, otherwise CPU
# This allows the model to run on GPU for faster training if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters for the model training
# These control the model architecture, training process, and data loading
in_channels = 1  # Number of input channels (1 for grayscale MNIST images)
num_classes = 10  # Number of output classes (digits 0-9 for MNIST)
learning_rate = 0.001  # Learning rate for the Adam optimizer (controls step size in gradient descent)
batch_size = 64  # Number of samples processed in each training batch
num_epochs = 5  # Number of complete passes through the training dataset
load_model = True  # Flag to determine whether to load a saved checkpoint at startup
# Load the training dataset
# MNIST dataset: 60,000 training images of handwritten digits (28x28 pixels)
# transform.ToTensor() converts PIL images to PyTorch tensors and scales pixel values to [0,1]
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
# Create data loader for training data
# shuffle=True randomizes the order of samples in each epoch to improve training
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Load the test dataset
# MNIST dataset: 10,000 test images for evaluation
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
# Create data loader for test data
# Note: shuffle=True is used here for consistency, but typically test data doesn't need shuffling
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the convolutional neural network model and move it to the specified device
# This creates an instance of our CNN and transfers it to GPU if available
model = ConvolutionalNeuralNetwork().to(device=device)

# Define loss function and optimizer
# CrossEntropyLoss combines LogSoftmax and NLLLoss - suitable for multi-class classification
criterion = nn.CrossEntropyLoss()
# Adam optimizer: adaptive learning rate optimization algorithm
# model.parameters() provides all trainable parameters of the model
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to save model checkpoint to disk
# Checkpoints include both model weights and optimizer state for complete training resumption
def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('=> Saving checkpoint...')
    # torch.save() serializes the state dictionary to a file
    # .pth.tar is a common extension for PyTorch checkpoints
    torch.save(state, filename)

# Function to load model checkpoint from disk
# This restores both model weights and optimizer state
def load_checkpoint(checkpoint):
    print('=> Loading Checkpoint...')
    # Load the saved state dictionary into the model
    model.load_state_dict(checkpoint['state_dict'])
    # Load the optimizer state (learning rate, momentum, etc.)
    optimizer.load_state_dict(checkpoint['optimizer'])


# Load checkpoint if load_model flag is True
# This allows resuming training from a previously saved state
if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'))

# Training loop: iterate over epochs and batches
for epoch in range(num_epochs):  # Loop over each training epoch
    # Save checkpoint every 3 epochs to preserve training progress
    if epoch % 3 == 0:
        # Create checkpoint dictionary containing model and optimizer state
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
    
    # Loop over batches in the training data
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move data and targets to the device (GPU/CPU)
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward pass: compute model predictions (scores/logits)
        scores = model(data)
        # Compute loss between predictions and true labels
        loss = criterion(scores, targets)

        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear previous gradients to prevent accumulation
        loss.backward()  # Backpropagate the loss to compute gradients

        # Update model parameters using computed gradients
        optimizer.step()

# Function to check model accuracy on a given dataset
# This evaluates the model performance without updating weights
def check_accuracy(loader, model):
    # Determine if we're checking training or test data for appropriate messaging
    if loader.dataset.train:  # Check the train attribute of the dataset
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    
    num_correct = 0  # Counter for correct predictions
    num_samples = 0  # Counter for total samples processed
    model.eval()  # Set model to evaluation mode (disables dropout, batch norm, etc.)

    # Disable gradient computation for efficiency during evaluation
    with torch.no_grad():
        for x, y in loader:  # Iterate over batches
            x = x.to(device)  # Move input data to device
            y = y.to(device)  # Move labels to device

            scores = model(x)  # Forward pass to get predictions
            # Get predicted class by finding index of maximum score
            _, predictions = scores.max(1)
            # Count correct predictions (element-wise comparison)
            num_correct += (predictions == y).sum()
            # Count total samples in this batch
            num_samples += predictions.size(0)

        # Calculate and print accuracy percentage
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples) * 100:.2f}%')
    # Set model back to training mode for potential further training
    model.train()

# Evaluate model performance on both training and test datasets
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

