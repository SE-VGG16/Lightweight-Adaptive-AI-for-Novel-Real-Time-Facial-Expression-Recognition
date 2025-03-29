# preprocessing.py
import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FERDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with image paths and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = int(self.annotations.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample

if __name__ == "__main__":
    # Quick test for dataset loading
    dataset = FERDataset(csv_file='data/annotations.csv', root_dir='data/images')
    print("Dataset size:", len(dataset))
    sample = dataset[0]
    print("Image shape:", sample['image'].shape, "Label:", sample['label'])
