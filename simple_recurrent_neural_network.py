# Import necessary libraries for PyTorch deep learning
import torch  # Main PyTorch library
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import torch.nn.functional as F  # Functional interface for neural network operations
from torch.utils.data import DataLoader  # Data loading utilities
import torchvision.datasets as datasets  # Pre-built datasets
import torchvision.transforms as transforms  # Image transformations

# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters for the model
input_size = 28  # Input size per time step (28 pixels per row in MNIST)
sequence_length = 28  # Number of time steps (28 rows in MNIST image)
num_layers = 2  # Number of RNN layers
hidden_size = 256  # Number of hidden units in RNN
num_classes = 10  # Number of output classes (digits 0-9)
learning_rate = 0.001  # Learning rate for optimizer
batch_size = 64  # Number of samples per batch
num_epochs = 2  # Number of training epochs

# Define a recurrent neural network class for sequence classification
class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RecurrentNeuralNetwork, self).__init__()  # Call parent class constructor
        self.hidden_size = hidden_size  # Store hidden size for later use
        self.num_layers = num_layers  # Store number of layers
        # RNN layer: processes sequences with specified input size, hidden size, and layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Fully connected layer: input is hidden_size * sequence_length, output is num_classes
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Initialize hidden state: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)
        # Forward pass through RNN: returns output and final hidden state (we ignore hidden state with _)
        out, _ = self.rnn(x, h0)
        # Reshape output to flatten for fully connected layer: (batch_size, hidden_size * sequence_length)
        out = out.reshape(out.shape[0], -1)
        # Apply fully connected layer (no activation, handled by loss)
        out = self.fc(out)
        return out



# Load the training dataset
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
# Create data loader for training data
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Load the test dataset
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
# Create data loader for test data (note: variable name should be test_loader, not train_loader)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the recurrent neural network model and move to device
model = RecurrentNeuralNetwork(input_size, hidden_size, num_layers, num_classes).to(device=device) 

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Train the network
for epoch in range(num_epochs):  # Loop over epochs
    for batch_idx, (data, targets) in enumerate(train_loader):  # Loop over batches
        # Move data and targets to the device (GPU/CPU)
        data = data.to(device=device).squeeze(1)  # Squeeze to remove channel dim: (batch, 1, 28, 28) -> (batch, 28, 28)
        targets = targets.to(device=device)



        # Forward pass: compute predictions
        scores = model(data)
        # Compute loss
        loss = criterion(scores, targets)

        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagate the loss

        # Update model parameters
        optimizer.step()

# Function to check accuracy on a dataset
def check_accuracy(loader, model):
    if loader.dataset.train:  # Check if the loader is for training data
        print("Checking accuracy on training data")  # Print message for training data
    else:
        print("Checking accuracy on test data")  # Print message for test data
    num_correct = 0  # Counter for correct predictions
    num_samples = 0  # Counter for total samples
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        for x, y in loader:  # Loop over batches
            x = x.to(device).squeeze(1)  # Move data to device and squeeze channel dim
            y = y.to(device)  # Move labels to device

            scores = model(x)  # Forward pass
            _, predictions = scores.max(1)  # Get predicted class (index of max score)
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

        # Print accuracy
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples) * 100:.2f}%')
    model.train()  # Set model back to training mode after evaluation
    

# Check accuracy on training and test sets
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

