import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from datasets.ton_iot_loader import ToN_IoT # for tabular data

class _ImageFeatureExtractor(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3)
        )
        if input_channels == 1:
            self._feature_dim = 64 * 3 * 3
        else:
            self._feature_dim = 64 * 4 * 4

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).view(x.size(0), -1)

class _TabularFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self._feature_dim = 32

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar', 'ton_iot'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for {args.dataset}")

    if args.dataset == 'mnist':
        extractor_core = _ImageFeatureExtractor(input_channels=1, num_classes=10)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        save_path = "pretrained_mnist_extractor.pth"
    elif args.dataset == 'cifar':
        extractor_core = _ImageFeatureExtractor(input_channels=3, num_classes=10)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        save_path = "pretrained_cifar_extractor.pth"
    elif args.dataset == 'ton_iot':
        extractor_core = _TabularFeatureExtractor(input_dim=10)
        train_dataset, _ = ToN_IoT()
        save_path = "pretrained_ton_iot_extractor.pth"

    classifier = nn.Linear(extractor_core.feature_dim, 10)
    full_model = nn.Sequential(extractor_core, classifier).to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

    optimizer = optim.Adam(full_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 3
    print(f"Starting rapid pre-training for {epochs} epochs...")
    full_model.train()
    
    for ep in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = full_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 99:
                print(f"  Epoch [{ep+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {total_loss/100:.4f}, Acc: {100.*correct/total:.2f}%")
                total_loss = 0.0

    print("Pre-training complete.")
    torch.save(extractor_core.state_dict(), save_path)
    print(f"\n[SUCCESS] Saved frozen feature extractor weights to: {save_path}")

if __name__ == '__main__':
    main()
