import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# -------------------------------
# CONFIGURATION
# -------------------------------
DATA_DIR = "training"  # Your root dir with 'flat/' and 'pitched/'
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.001
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# TRANSFORMS
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # RGB normalization
])

# -------------------------------
# DATASET AND LOADERS
# -------------------------------
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# FINAL CNN MODEL
# -------------------------------
class RoofClassifierCNN(nn.Module):
    def __init__(self):
        super(RoofClassifierCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 16x64x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 32x32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 64x16x16
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x

model = RoofClassifierCNN().to(DEVICE)

# -------------------------------
# LOSS & OPTIMIZER
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------------
# TRAINING LOOP
# -------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1:02d}/{EPOCHS} - Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")

# -------------------------------
# VALIDATION
# -------------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

val_acc = 100 * correct / total
print(f"\n✅ Final Validation Accuracy: {val_acc:.2f}%")

# -------------------------------
# SAVE MODEL
# -------------------------------
torch.save(model.state_dict(), "roof_type_cnn_best.pth")
print("🧠 Model saved as roof_type_cnn_best.pth")
