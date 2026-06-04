from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class ImageDataset(Dataset):
    def __init__(self, image_path, normalize=True, grayscale=True):
        # Load image, convert to grayscale and scale pixel values to [0, 1]
        self.image = Image.open(image_path)
        if grayscale:
            self.image = self.image.convert('L')
        self.grayscale = grayscale
        self.image = ToTensor()(self.image)
        self.normalize = normalize
        self.image = torch.flip(self.image, [1])  # flip along height dimension

        # Get image dimensions
        self.height, self.width = self.image.shape[1:]

    def __len__(self):
        return self.height * self.width

    def __getitem__(self, idx):
        # Convert flat index to 2D coordinates
        row = idx // self.width
        col = idx % self.width

        # Scale coordinates to [-1, 1]
        if self.normalize:
            input = torch.tensor([col / (self.width / 2) - 1, (self.height-row) / (self.height / 2) - 1])
        else:
            input = torch.tensor([col, self.height-row], dtype=torch.float32)
        input = torch.tensor([col / (self.width / 2) - 1, row / (self.height / 2) - 1])

        # Get pixel value
        if self.grayscale:
            output = self.image[0, row, col]
        else:
            output = self.image[:, row, col]

        return input, output
    
    def display_image(self):
        # Check the shape of the image tensor and handle accordingly
        image = self.image
        image = torch.flip(image, [1])
        if len(image.shape) == 3:  # For RGB images (C, H, W)
            plt.imshow(image.numpy().transpose(1, 2, 0))
        elif len(image.shape) == 2:  # For grayscale images (H, W)
            plt.imshow(image.numpy(), cmap='gray')
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        plt.axis('off')
        plt.show()

