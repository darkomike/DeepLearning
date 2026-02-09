import torch

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('=> Saving checkpoint...')
    # torch.save() serializes the state dictionary to a file
    # .pth.tar is a common extension for PyTorch checkpoints
    torch.save(state, filename)

# Function to load model checkpoint from disk
# This restores both model weights and optimizer state
def load_checkpoint(checkpoint,model,optimizer):
    print('=> Loading Checkpoint...')
    # Load the saved state dictionary into the model
    model.load_state_dict(checkpoint['state_dict'])
    # Load the optimizer state (learning rate, momentum, etc.)
    optimizer.load_state_dict(checkpoint['optimizer'])
