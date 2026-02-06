# Import necessary libraries for PyTorch deep learning
import torch  # Main PyTorch library for tensor operations and neural networks
import torch.nn as nn  # Neural network modules (layers, activations, etc.)
import torch.optim as optim  # Optimization algorithms (Adam, SGD, etc.)
import torch.nn.functional as F  # Functional interface for operations like activation functions
from torch.utils.data import DataLoader  # Utilities for loading and batching data
import torchvision.datasets as datasets  # Pre-built datasets like MNIST
import torchvision.transforms as transforms  # Image transformations (ToTensor, etc.)



# Set the device to GPU if available, otherwise CPU
# This allows the model to run on GPU for faster training if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters for the model
# These control the architecture and training behavior of the neural network
input_size = 28  # Each time step processes 28 pixels (one row of the 28x28 MNIST image)
sequence_length = 28  # Total time steps: 28 rows in the image, processed sequentially
num_layers = 2  # Number of stacked LSTM layers for deeper representation learning
hidden_size = 256  # Number of hidden units in each LSTM layer (controls model capacity)
num_classes = 10  # Output classes: digits 0-9 for MNIST classification
learning_rate = 0.001  # Step size for optimizer updates (Adam default is 0.001)
batch_size = 64  # Number of samples processed together in each training step
num_epochs = 2  # Number of complete passes through the training dataset

# Define a simple bidirectional long short-term memory (LSTM) neural network class for sequence classification
# Bidirectional LSTM processes sequences in both forward and backward directions for richer context
class BidirectionalLongShortTermMemoryNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BidirectionalLongShortTermMemoryNeuralNetwork, self).__init__()  # Initialize parent nn.Module class
        self.hidden_size = hidden_size  # Store hidden size for initializing hidden states
        self.num_layers = num_layers  # Store number of layers for hidden state initialization
        # Bidirectional LSTM layer: processes input sequences in both directions
        # input_size: features per time step, hidden_size: hidden units, num_layers: stacked layers
        # batch_first=True: input shape is (batch, seq_len, input_size)
        # bidirectional=True: processes forward and backward, doubling hidden states
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        # Fully connected layer: maps LSTM output to class predictions
        # Input size is hidden_size * 2 because bidirectional doubles the hidden dimension
        # Output size is num_classes for classification
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Initialize hidden state for LSTM: shape (num_layers * 2, batch_size, hidden_size)
        # Multiplied by 2 for bidirectional (forward + backward directions)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device=device)
        # Initialize cell state for LSTM: same shape as hidden state
        # Cell state maintains long-term memory in LSTM
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device=device)
        # Forward pass through bidirectional LSTM
        # Input: (batch_size, seq_len, input_size), hidden/cell states
        # Output: (batch_size, seq_len, hidden_size * 2), final hidden/cell states (ignored with _)
        out, _ = self.lstm(x, (h0, c0))
        # Take the last time step's output for classification: (batch_size, hidden_size * 2)
        # This contains the final representation after processing the entire sequence
        out = out[:, -1, :]
        # Apply fully connected layer to get class scores: (batch_size, num_classes)
        # No activation here - CrossEntropyLoss handles softmax internally
        out = self.fc(out)
        return out



# Load the training dataset
# MNIST dataset: 60,000 training images of handwritten digits (28x28 pixels each)
# transform=transforms.ToTensor(): converts PIL images to PyTorch tensors and normalizes to [0,1]
# download=True: automatically downloads the dataset if not present
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)

# Create data loader for training data
# DataLoader batches the dataset and shuffles for better training
# batch_size: number of samples per batch, shuffle=True: randomize order each epoch
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Load the test dataset
# MNIST test set: 10,000 images for evaluating model performance
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
# Create data loader for test data
# Note: shuffle=True for test data is unusual (typically False), but kept as in original
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the bidirectional LSTM neural network model
# Pass hyperparameters to constructor and move model to specified device (GPU/CPU)
model = BidirectionalLongShortTermMemoryNeuralNetwork(input_size, hidden_size, num_layers, num_classes).to(device=device) 

# Define loss function and optimizer
# CrossEntropyLoss: combines softmax activation and negative log likelihood for multi-class classification
criterion = nn.CrossEntropyLoss()
# Adam optimizer: adaptive learning rate optimization algorithm
# lr=learning_rate: controls step size, model.parameters(): tensors to optimize
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop: iterate over epochs and batches to update model weights
for epoch in range(num_epochs):  # Loop over each training epoch
    for batch_idx, (data, targets) in enumerate(train_loader):  # Loop over batches in training data
        # Move input data and target labels to the specified device (GPU/CPU)
        # data shape: (batch_size, 1, 28, 28) - squeeze(1) removes channel dim → (batch_size, 28, 28)
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)  # targets shape: (batch_size,)

        # Forward pass: compute model predictions (scores) for the current batch
        scores = model(data)  # scores shape: (batch_size, num_classes)
        
        # Compute loss: measure difference between predictions and true labels
        loss = criterion(scores, targets)

        # Backward pass: compute gradients of loss with respect to model parameters
        optimizer.zero_grad()  # Reset gradients from previous batch to avoid accumulation
        loss.backward()  # Backpropagate: compute gradients through the network

        # Update model parameters using computed gradients
        optimizer.step()  # Apply optimizer step to update weights

# Function to evaluate model accuracy on a given dataset
# This function computes and prints the accuracy of the model on training or test data
def check_accuracy(loader, model):
    # Determine if we're checking training or test data and print appropriate message
    if loader.dataset.train:  # Check the train attribute of the dataset
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    
    # Initialize counters for correct predictions and total samples
    num_correct = 0  # Number of correctly classified samples
    num_samples = 0  # Total number of samples evaluated
    
    # Set model to evaluation mode: disables dropout, batch norm updates, etc.
    model.eval()
    
    # Disable gradient computation for efficiency during evaluation
    with torch.no_grad():
        # Iterate over batches in the data loader
        for x, y in loader:
            # Move input data and labels to the device
            x = x.to(device).squeeze(1)  # Remove channel dimension from images
            y = y.to(device)  # Ground truth labels
            
            # Forward pass: get model predictions
            scores = model(x)  # Raw class scores from model
            # Get predicted class by finding index of maximum score
            _, predictions = scores.max(1)  # predictions shape: (batch_size,)
            
            # Count correct predictions: compare predictions with true labels
            num_correct += (predictions == y).sum()  # Sum boolean tensor to get count
            # Count total samples in this batch
            num_samples += predictions.size(0)  # Batch size
    
    # Calculate and print accuracy as percentage
    accuracy = float(num_correct) / float(num_samples) * 100
    print(f'Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%')
    
    # Set model back to training mode for potential further training
    model.train()

# Evaluate model performance on training and test datasets
# This shows how well the model learned from training data and generalizes to unseen data
check_accuracy(train_loader, model)  # Check training accuracy
check_accuracy(test_loader, model)   # Check test accuracy

