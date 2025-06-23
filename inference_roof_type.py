import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = 128
MODEL_PATH = "roof_type_cnn_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['flat', 'pitched']  # Must match training order

# -------------------------------
# MODEL ARCHITECTURE (same as training)
# -------------------------------
class RoofClassifierCNN(nn.Module):
    def __init__(self):
        super(RoofClassifierCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
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
        return self.fc(x)

# -------------------------------
# TRANSFORMS
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------------------------------
# LOAD MODEL
# -------------------------------
model = RoofClassifierCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except:
        print(f"❌ Failed to open image: {image_path}")
        return

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        confidence = torch.softmax(outputs, dim=1)[0][class_idx].item()

    print(f"✅ Prediction: {CLASS_NAMES[class_idx]} (Confidence: {confidence*100:.2f}%)")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference_roof_type.py <image_path>")
    else:
        predict(sys.argv[1])
