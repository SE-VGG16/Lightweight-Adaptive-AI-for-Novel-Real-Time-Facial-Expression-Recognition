# evaluation.py
import torch
from torch.utils.data import DataLoader
from model import AMFT
from preprocessing import FERDataset
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, device, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues,
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = FERDataset(csv_file='data/test_annotations.csv', root_dir='data/test_images')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    num_classes = 7
    model = AMFT(num_classes=num_classes, use_transformer=True, use_lstm=False)
    model.to(device)
    
    # Load model checkpoint
    checkpoint_path = 'checkpoints/model_epoch_10.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    true_labels, preds = evaluate(model, device, test_loader)
    acc = accuracy_score(true_labels, preds)
    print("Test Accuracy: {:.2f}%".format(acc * 100))
    print("Classification Report:")
    print(classification_report(true_labels, preds))
    cm = confusion_matrix(true_labels, preds)
    class_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
    plot_confusion_matrix(cm, classes=class_names)

if __name__ == '__main__':
    main()
