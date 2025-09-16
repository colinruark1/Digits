from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.softmax(x)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\t{loss.item():.6f}')

def test():
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders['test'].dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f})')



if __name__ == "__main__":
    train_data = datasets.MNIST(
        root = 'data',
        train = True,
        transform = ToTensor(),
        download = True
    )

    test_data = datasets.MNIST(
        root = 'data',
        train = False,
        transform = ToTensor(),
        download = True
    )

    loaders = {
        'train': DataLoader(train_data,
                            batch_size=100,
                            shuffle=True,
                            num_workers=1),

        'test': DataLoader(test_data,
                            batch_size=100,
                            shuffle=True,
                            num_workers=1),
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_fn = nn.CrossEntropyLoss()
    print(train_data)
    print(test_data)
    print(train_data.data.size())
    print(train_data.targets)
    print(loaders)

    for epoch in range(1, 11):
        train(epoch)
        test()

    # Assuming you already have your trained model and test dataset
    model.eval()  # Set the model to evaluation mode

    # Get a batch of test data
    test_loader = DataLoader(
        test_data,
        batch_size=64,  # Set batch size (e.g., 64 samples per batch)
        shuffle=False   # No need to shuffle test data
    )

    test_loader_iter = iter(test_loader)
    images, labels = next(test_loader_iter)

    # Pass the images through the model to get predictions
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Find misclassified samples
    misclassified = (predicted != labels).nonzero(as_tuple=True)[0]

    # Pick one misclassified example
    if len(misclassified) > 0:
        rand = random.randint(0, len(misclassified) - 1)
        index = misclassified[0].item()
        image = images[index].squeeze().cpu().numpy()  # Convert to numpy for plotting
        true_label = labels[index].item()
        predicted_label = predicted[index].item()

        # Plot the image
        plt.imshow(image, cmap='gray')
        plt.title(f"True Label: {true_label}, Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()
        print(len(misclassified))
    else:
        print("No misclassified examples in this batch!")


