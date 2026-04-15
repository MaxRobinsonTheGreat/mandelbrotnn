import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

from src.models import Simple, SkipConn, Fourier, CenteredLinearMap

# Hyperparameters and constants
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
MOMENTUM = 0.9
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data loading and preprocessing
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
eval_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# save 10 images from the dataset in the folder captures/mnist, create the folder first

os.makedirs('./captures/mnist', exist_ok=True)
for i in range(10):
    image, label = train_dataset[i]
    plt.imsave(f'./captures/mnist/{i}.png', image.squeeze().numpy(), cmap='gray')

normalizer = 1
model = SkipConn(init_size=28*28, final_size=10).to(DEVICE)
# model = Fourier(init_size=28*28, final_size=10, fourier_order=8).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

# Lists to keep track of metrics
train_losses = []
eval_losses = []
eval_accuracies = []

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        data = data.view(data.size(0), -1)  # Flatten the data
        optimizer.zero_grad()
        data *= normalizer
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Evaluation
    model.eval()
    eval_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            data = data.view(data.size(0), -1)
            data *= normalizer
            output = model(data)
            eval_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_eval_loss = eval_loss / len(eval_loader)
    eval_losses.append(avg_eval_loss)

    accuracy = 100. * correct / len(eval_loader.dataset)
    eval_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{EPOCHS}\tTrain Loss: {avg_train_loss:.4f}\tEval Loss: {avg_eval_loss:.4f}\tAccuracy: {accuracy:.2f}%")

# Plotting
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(eval_losses, label='Evaluation Loss')
plt.title('Loss over time')
plt.xlabel('Training Time')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(eval_accuracies)
plt.title('Evaluation Accuracy over time')
plt.xlabel('Training Time')
plt.ylabel('Accuracy (%)')
# set y axis limits to 0 and 100
plt.ylim(0, 100)


plt.tight_layout()
plt.show()

display_image_indices = [0,1,2]
# Predict and plot each image
for i, index in enumerate(display_image_indices, 1):
    image, label = eval_dataset[index]
    image = image.to(DEVICE).view(1, -1)  # Add batch dimension and flatten
    image *= normalizer
    output = model(image)
    pred = output.argmax(dim=1).item()

    plt.subplot(1, 3, i)
    plt.imshow(image.view(28, 28).cpu().numpy(), cmap='gray')
    plt.title(f"True label: {label}\nPredicted label: {pred}")
    plt.axis('off')

plt.tight_layout()
plt.show()