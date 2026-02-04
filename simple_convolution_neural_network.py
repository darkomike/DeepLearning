# Import necessary libraries for PyTorch deep learning
import torch  # Main PyTorch library
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import torch.nn.functional as F  # Functional interface for neural network operations
from torch.utils.data import DataLoader  # Data loading utilities
import torchvision.datasets as datasets  # Pre-built datasets
import torchvision.transforms as transforms  # Image transformations

# Define a convolutional neural network class for image classification
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ConvolutionalNeuralNetwork, self).__init__()  # Call parent class constructor
        # First convolutional layer: 1 input channel (grayscale), 8 output channels, 3x3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # Max pooling layer: 2x2 kernel, stride 2 (reduces spatial dimensions by half)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        # Second convolutional layer: 8 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # Fully connected layer: input size 16*7*7 (after convolutions and pooling), output num_classes
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        # Apply first convolution and ReLU activation
        x = F.relu(self.conv1(x))
        # Apply max pooling
        x = self.pool(x)
        # Apply second convolution and ReLU activation
        x = F.relu(self.conv2(x))
        # Apply max pooling again
        x = self.pool(x)
        # Flatten the tensor for the fully connected layer
        x = x.reshape(x.shape[0], -1)
        # Apply fully connected layer (no activation, handled by loss)
        x = self.fc1(x)
        return x

# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters for the model
in_channels = 1  # Number of input channels (1 for grayscale MNIST images)
num_classes = 10  # Number of output classes (digits 0-9)
learning_rate = 0.001  # Learning rate for optimizer
batch_size = 64  # Number of samples per batch
num_epochs = 5  # Number of training epochs

# Load the training dataset
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
# Create data loader for training data
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Load the test dataset
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
# Create data loader for test data (note: variable name should be test_loader, not train_loader)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the convolutional neural network model and move to device
model = ConvolutionalNeuralNetwork().to(device=device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Train the network
for epoch in range(num_epochs):  # Loop over epochs
    for batch_idx, (data, targets) in enumerate(train_loader):  # Loop over batches
        # Move data and targets to the device (GPU/CPU)
        data = data.to(device=device)
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
            x = x.to(device)  # Move data to device
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

