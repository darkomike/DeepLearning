# Import necessary libraries for PyTorch deep learning
import torch  # Main PyTorch library
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import torch.nn.functional as F  # Functional interface for neural network operations
from torch.utils.data import DataLoader  # Data loading utilities
import torchvision.datasets as datasets  # Pre-built datasets
import torchvision.transforms as transforms  # Image transformations
import torchvision
# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters for the model
in_channel = 3
num_classes = 10  # Number of output classes (digits 0-9)
learning_rate = 0.001  # Learning rate for optimizer
batch_size = 1024  # Number of samples per batch
num_epochs = 5  # Number of training epochs

# Load pretrain model and modify it
# Create an Identity layer to replace the adaptive average pooling layer
# This allows us to use the feature maps directly without spatial reduction
class Identity(nn.Module):
    """
    Identity layer that passes input unchanged.
    Used to replace pooling layers in pre-trained models.
    """
    def __init__(self):
        super(Identity,self).__init__()

    def forward(self,x):
        return x

# Load pre-trained VGG16 model with ImageNet weights
model = torchvision.models.vgg16(pretrained=True)

# Freeze all pre-trained parameters to prevent them from being updated during training
# This preserves the learned features from ImageNet
for param in model.parameters():
    param.requires_grad = False

# Replace the adaptive average pooling with Identity to maintain feature map dimensions
# This allows us to use the full spatial information from the feature extractor
model.avgpool = Identity()

# Replace the classifier with a custom network that includes flattening
# VGG16 features output: (batch, 512, 7, 7) -> flatten to (batch, 512*7*7 = 25088)
# But we'll use a simpler approach: add flattening in the sequential
model.classifier = nn.Sequential(
    nn.Flatten(),        # Flatten: (batch, 512, 7, 7) -> (batch, 25088)
    nn.Linear(512*7*7, 100),  # First FC: 25088 -> 100
    nn.ReLU(),           # Activation
    nn.Linear(100, 10)   # Output: 100 -> 10 classes
)

# Move model to the appropriate device (GPU/CPU)
model.to(device)
# Load the training dataset
# CIFAR-10 consists of 60,000 32x32 color images in 10 classes
train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
# Create data loader for training data with shuffling for randomness
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Load the test dataset (10,000 images for evaluation)
test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
# Create data loader for test data (note: variable name should be test_loader, not train_loader)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the GRU neural network model and move to device

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Train the network
for epoch in range(num_epochs):  # Loop over epochs
    for batch_idx, (data, targets) in enumerate(train_loader):  # Loop over batches
        # Move data and targets to the device (GPU/CPU)
        # CIFAR-10 images are 32x32x3, so no squeeze needed unlike MNIST
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
    """
    Evaluate model accuracy on a given dataset.
    
    Args:
        loader: DataLoader for the dataset to evaluate
        model: The neural network model to evaluate
    """
    if loader.dataset.train:  # Check if the loader is for training data
        print("Checking accuracy on training data")  # Print message for training data
    else:
        print("Checking accuracy on test data")  # Print message for test data
    num_correct = 0  # Counter for correct predictions
    num_samples = 0  # Counter for total samples
    model.eval()  # Set model to evaluation mode (disables dropout, batch norm)

    with torch.no_grad():  # Disable gradient computation for efficiency
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

