# 

import h5py
import torch
import os
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from ddpm_config import img_size 

class H5ArrayDataset(Dataset):
    def __init__(self, h5_path, dataset_name):
        self.h5_path = h5_path
        self.dataset_name = dataset_name

        # Open the HDF5 file once and keep it open
        self.hf = h5py.File(h5_path, 'r')
        self.dataset = self.hf[dataset_name]
        self.len = self.dataset.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        item = self.dataset[index]  # shape: (H, W, C)
        h5_tensor = torch.from_numpy(item.astype('float32'))
        h5_tensor = h5_tensor.permute(2, 0, 1)  # CHW format
        return h5_tensor

    def __del__(self):
        # Close the HDF5 file when Dataset is destroyed
        if hasattr(self, 'hf') and self.hf:
            self.hf.close()

class MCImageDataset(Dataset):
    def __init__(self, images_folder):
        self.images_folder = os.path.join(os.getcwd(), images_folder)
        self.images = [f for f in os.listdir(self.images_folder) if f.lower().endswith(('png', 'jpg'))]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(os.getcwd(), self.images_folder, self.images[index])
        image = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5,))])
        image_tensor = transform(image)  # Shape: (C, H, W)
        return image_tensor
    