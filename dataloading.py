import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms


class MCImageDataset(Dataset):
    def __init__(self, images_folder):
        self.images_folder = os.path.join(os.getcwd(), images_folder)
        self.images = [f for f in os.listdir(self.images_folder) if f.lower().endswith(('png', 'jpg'))]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(os.getcwd(), self.images_folder, self.images[index])
        image = Image.open(img_path).convert('L')
        transform = transforms.Compose([
                                transforms.Resize((128,128)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5,))])
        image_tensor = transform(image)  # Shape: (C, H, W)
        return image_tensor
    