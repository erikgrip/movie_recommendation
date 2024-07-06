import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MovieRecommendationModel
from dataset import MovieDataset

# Set random seed for reproducibility
torch.manual_seed(42)

# Load the dataset
dataset = MovieDataset("data/dataset.zip")

# Create the model
model = MovieRecommendationModel()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(10):
    running_loss = 0.0
    for batch in dataloader:
        inputs, labels = batch

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the average loss for the epoch
    print(f"Epoch {epoch+1}: Loss = {running_loss / len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), "trained_model.pth")