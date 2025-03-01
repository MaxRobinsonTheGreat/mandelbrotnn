from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class ImageDataset(Dataset):
    def __init__(self, image_path):
        # Load image, convert to grayscale and scale pixel values to [0, 1]
        self.image = Image.open(image_path).convert('L')
        self.image = ToTensor()(self.image)
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
        input = torch.tensor([col / (self.width / 2) - 1, row / (self.height / 2) - 1])

        # Get pixel value
        output = self.image[0, row, col]

        return input, output
    
    def display_image(self):
        # uses the getitem method to get each pixel value and displays the final image. used for debugging
        image = torch.zeros((self.height, self.width))
        for i in range(len(self)):
            row = i // self.width
            col = i % self.width
            image[row, col] = self[i][1]
        plt.imshow(image, cmap='gray')
        plt.show()

