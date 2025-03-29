# test.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import AMFT
from preprocessing import FERDataset

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dataset and DataLoader
    test_dataset = FERDataset(csv_file='data/test_annotations.csv', root_dir='data/test_images')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Model
    num_classes = 7
    model = AMFT(num_classes=num_classes, use_transformer=True, use_lstm=False)
    model.to(device)
    
    # Load model checkpoint (adjust the path as needed)
    checkpoint_path = 'checkpoints/model_epoch_10.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    test(model, device, test_loader, criterion)

if __name__ == '__main__':
    main()
