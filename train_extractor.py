import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class _ImageFeatureExtractor(nn.Module):
    """Exactly mirrors the _ImageFeatureExtractor in server/semantic_analyzer.py.
    
    Lightweight CNN feature extractor for MNIST. The last classification layer is removed,
    exposing the penultimate representation.
    """
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5), # features.0
            nn.ReLU(),                       # features.1
            nn.MaxPool2d(2),                 # features.2
            nn.Conv2d(16, 32, kernel_size=3),# features.3
            nn.ReLU(),                       # features.4
            nn.MaxPool2d(2),                 # features.5
            nn.Conv2d(32, 64, kernel_size=3) # features.6
        )
        self._feature_dim = 64 * 3 * 3

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).view(x.size(0), -1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build the full model with a temporary classifier head
    extractor_core = _ImageFeatureExtractor(input_channels=1, num_classes=10)
    classifier = nn.Linear(extractor_core.feature_dim, 10)
    full_model = nn.Sequential(extractor_core, classifier).to(device)

    # Load clean, unpoisoned baseline MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Downloading/Loading MNIST dataset...")
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2 if device.type == 'cuda' else 0)

    # Fast Training logic
    optimizer = optim.Adam(full_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 3
    print(f"Starting rapid pre-training for {epochs} epochs...")
    full_model.train()
    
    for ep in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = full_model(images)
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
    
    # Save STRICTLY the extractor core's state dictionary to avoid mismatched shape errors on load
    save_path = "pretrained_mnist_extractor.pth"
    torch.save(extractor_core.state_dict(), save_path)
    print(f"\n[SUCCESS] Saved frozen feature extractor weights to: {save_path}")
    print("You can now safely start the LUP-SAC FL pipeline!")

if __name__ == '__main__':
    main()
