import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchtext.datasets import Multi30k
from torchtext.data import Field , BucketIterator 
import numpy as np
import spacy 
import random
from torch.utils.tensorboard import SummaryWriter 
from utils import load_checkpoint,save_checkpoint, blue,translate_sentence

# Load spaCy models for German and English tokenization
# These models provide pre-trained tokenizers for natural language processing
spacy_gen = spacy.load('de')  # German language model
spacy_eng = spacy.load('en')  # English language model

# Define tokenizer functions using spaCy
# These functions convert text sentences into lists of tokens (words/subwords)
def tokenizer_ger(text):
    # Tokenize German text and return list of token texts
    return [tok.text for tok in spacy_gen.tokenizer(text)]

def tokenizer_eng(text):
    # Tokenize English text and return list of token texts
    return [tok.text for tok in spacy_eng.tokenizer(text)]

# Define Field objects for preprocessing text data
# Fields specify how to process and numericalize text sequences
german = Field(tokenizer=tokenizer_ger, lower=True, init_token='<sos>', eos_token='<eos>')
# - tokenizer: function to split text into tokens
# - lower: convert to lowercase
# - init_token: start-of-sequence token
# - eos_token: end-of-sequence token

english = Field(tokenizer=tokenizer_eng, lower=True, init_token='<sos>', eos_token='<eos>')

# Load the Multi30k dataset for German-English translation
# This dataset contains sentence pairs for machine translation training
train_data, validation_data, test_data = Multi30k.splits(exts=('.de','.en'),fields=(german,english))

# Build vocabularies for both languages
# Vocabularies map tokens to integer indices for neural network input
german.build_vocab(train_data,max_size=1000,min_freq=2)
# - max_size: maximum vocabulary size (most frequent tokens)
# - min_freq: minimum frequency for token inclusion

english.build_vocab(train_data,max_size=1000,min_freq=2)


# training paratmeters 
num_epochs = 20  # Number of complete passes through the training dataset
learning_rate = 0.001  # Step size for optimizer updates (controls learning speed)
batch_size = 64  # Number of samples processed before updating model parameters

# Model hyperparameters defining the neural network architecture
load_model = False  # Whether to load a pre-trained model checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
input_size_encoder = len(german.vocab)  # Vocabulary size for German (encoder input)
input_size_decoder = len(english.vocab)  # Vocabulary size for English (decoder input/output)
output_size = len(english.vocab)  # Output vocabulary size (same as decoder input)
encoder_embedding_size = 300  # Dimensionality of word embeddings for encoder
decoder_embedding_size = 300  # Dimensionality of word embeddings for decoder
hidden_size = 1024  # Number of hidden units in LSTM layers
num_layers = 2  # Number of stacked LSTM layers
enc_dropout = 0.5  # Dropout probability for encoder (prevents overfitting)
dec_dropout = 0.5  # Dropout probability for decoder




class Encoder(nn.Module):
    """
    Encoder component of the sequence-to-sequence model.
    Processes the source sequence (German) and produces context vectors.
    """
    def __init__(self, input_size, embedding_size,hidden_size,num_layers,p):
        super(Encoder,self).__init__()
        self.input_size = input_size  # Vocabulary size
        self.num_layers = num_layers  # Number of LSTM layers

        # Dropout layer to prevent overfitting by randomly zeroing some activations
        self.dropout = nn.Dropout(p)
        # Embedding layer converts token indices to dense vectors
        self.embedding = nn.Embedding(input_size, embedding_size)
        # LSTM layer for sequence processing
        self.rnn = nn.LSTM(embedding_size,hidden_size,num_layers,dropout=p)

    def forward(self,x):
        # x shape: (seq_lenght, N) where N is batch size
        # Apply dropout to embeddings for regularization
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_lenght, N, embedding_size)

        # Pass through LSTM: returns outputs and final hidden/cell states
        # We only need the final hidden and cell states for the decoder
        _,(hidden,cell) = self.rnn(embedding)

        # Return context vectors (hidden and cell states) for decoder initialization
        return hidden,cell

class Decoder(nn.Module):
    """
    Decoder component of the sequence-to-sequence model.
    Generates the target sequence (English) token by token.
    """
    def __init__(self, input_size, embedding_size,hidden_size,output_size,num_layers,p):
        super(Decoder,self).__init__()
        self.input_size = input_size  # Vocabulary size
        self.num_layers = num_layers  # Number of LSTM layers

        # Dropout for regularization
        self.dropout = nn.Dropout(p)
        # Embedding layer for target language tokens
        self.embedding = nn.Embedding(input_size, embedding_size)
        # LSTM layer for sequence generation
        self.rnn = nn.LSTM(embedding_size,hidden_size,num_layers,dropout=p)
        # Fully connected layer to predict next token probabilities
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x,hidden,cell):
        # x shape: (N) where N is batch size, but we need (1, N) for single token
        x = x.unsqueeze(0)  # Add sequence dimension: (1, N)

        # Embed the input token
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        # Pass through LSTM with previous hidden and cell states
        outputs, (hidden,cell) = self.rnn(embedding,(hidden,cell))
        # outputs shape: (1, N, hidden_size)

        # Project to vocabulary size for token prediction
        predictions = self.fc(outputs)
        # predictions shape: (1, N, length_of_vocab)

        # Remove sequence dimension for output
        predictions = predictions.squeeze(0)  # (N, vocab_size)

        return predictions,hidden,cell


class SequenceToSequence(nn.Module):
    """
    Complete sequence-to-sequence model combining encoder and decoder.
    Implements the full translation pipeline with teacher forcing.
    """
    def __init__(self, encoder,decoder):
        super(SequenceToSequence,self).__init__()
        self.encoder = encoder  # Encoder network
        self.decoder = decoder  # Decoder network

    def forward(self,source,target, teacher_force_ratio=0.5):
        # Get batch dimensions
        batch_size = source.shape[1]  # N
        target_len = target.shape[0]  # Length of target sequence
        target_vocab_size = len(english.vocab)  # Output vocabulary size

        # Initialize output tensor to store predictions for each target position
        outputs = torch.zeros(target_len, batch_size,target_vocab_size).to(device=device)

        # Encode the source sequence to get context vectors
        hidden,cell = self.encoder(source)

        # Start decoding with the <sos> token
        x = target[0]  # First token is always <sos>

        # Generate sequence token by token
        for t in range (1, target_len):  # Start from 1 to skip <sos>
            # Decode one token
            output, hidden,cell = self.decoder(x,hidden,cell)
            # Store prediction for this position
            outputs[t] = output

            # Get the best prediction (greedy decoding)
            best_guess = output.argmax(1)

            # Teacher forcing: use ground truth or model prediction randomly
            # This helps training stability and convergence
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
    

#tensorboard
writer = SummaryWriter('runs/loss_plot')  # TensorBoard writer for logging training metrics
step = 0  # Global step counter for TensorBoard

# Create data iterators for batching and efficient data loading
# BucketIterator groups sequences of similar lengths for better efficiency
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data,validation_data, test_data), 
    batch_size=batch_size,  # Number of sequences per batch
    sort_within_batch=True,  # Sort sequences within batch by length
    sort_key = lambda x: len(x.src),  # Sort batches by source sequence length
    device=device  # Move batches to GPU/CPU
)

# Initialize encoder and decoder networks
encoder_net = Encoder(input_size_encoder, encoder_embedding_size,hidden_size,num_layers,enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size,hidden_size,output_size,num_layers,dec_dropout).to(device)

# Create the complete sequence-to-sequence model
model = SequenceToSequence(encoder_net,decoder_net).to(device)

# Get the padding token index for loss function (ignore padding in loss calculation)
pad_idx = english.vocab.stoi['<pad>']
# Cross-entropy loss with padding mask
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
# Adam optimizer with weight decay for regularization
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.1)

# Load checkpoint if specified
if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'),model,optimizer)


#training 

for epoch in range(num_epochs):
    print(f'For epoch {epoch} / {num_epochs}')

    # Save model checkpoint at the start of each epoch
    checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
    save_checkpoint(checkpoint)
    
    # Iterate over training batches
    for batch_idx, batch in enumerate(train_iterator):
        # Get source (German) and target (English) sequences
        inp_data = batch.src.to(device)  # Source sequence
        target = batch.trg.to(device)    # Target sequence

        # Forward pass: generate predictions
        output = model(inp_data,target)
        # output shape: (trg_len, batch_size, output_dim)

        # Reshape for loss calculation (flatten batch and sequence dimensions)
        # We only compute loss on the first token prediction for simplicity
        output = output[:1].reshape(-1,output.shape[2])  # (batch_size, vocab_size)
        target = target[:1].reshape(-1)  # (batch_size,)

        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Compute cross-entropy loss
        loss = criterion(output,target)
        # Backpropagate gradients
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        # Update model parameters
        optimizer.step()

        # Log loss to TensorBoard
        writer.add_scalar('Training loss',loss,global_step=step)
        step += 1


