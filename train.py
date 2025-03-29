# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import AMFT
from preprocessing import FERDataset
import os

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch} Average Loss: {avg_loss:.4f}')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    num_classes = 7

    # Dataset and DataLoader
    train_dataset = FERDataset(csv_file='data/train_annotations.csv', root_dir='data/train_images')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Model
    model = AMFT(num_classes=num_classes, use_transformer=True, use_lstm=False)
    model.to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, criterion, optimizer, epoch)
        # Save checkpoint after each epoch
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch}.pth')

if __name__ == '__main__':
    main()
