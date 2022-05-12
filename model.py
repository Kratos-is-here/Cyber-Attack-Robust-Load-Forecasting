import pandas as pd
import random
import numpy as np
import torch
import time as t
import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  
from models import BRNN
from torch.utils.tensorboard import SummaryWriter
from BelgiumGridData import load_data
writer = SummaryWriter()
writer2 = SummaryWriter()
writer3 = SummaryWriter()
writer4 = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64

model = nn.Sequential(
    nn.Linear(2,50),
    nn.Linear(50,1),
    nn.Sigmoid(),
)

bidirectional_lstm_model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)

print(device)
print(model)

criterion = nn.L1Loss()
optim = optim.SGD(model.parameters(), lr = 0.01)
optim2 = optim.Adam(bidirectional_lstm_model.parameters(), lr = 0.001)

params = {
    'batch_size' : 32,
}

def run_me(model, data, test_data, params, criterion, optimizer, scaler = writer):
    
    train_loader = data(data)
    test_loader = data(test_data)
    

    num_epochs = 2

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)
            # plot the loss in tensorboard
            scaler.add('training-loss', loss.item(), epoch)
            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy  \
            {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()

run_me(model, load_data, load_data, params, criterion, optim)
run_me(bidirectional_lstm_model, load_data, load_data, params, criterion, optim2)
