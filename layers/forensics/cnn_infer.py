import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# modified for better path handling
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "casia_cnn.pth"
)
# MODEL_PATH = "casia_cnn.pth"  # Ensure this path is correct and the model file exists

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
# model.load_state_dict(torch.load("casia_cnn.pth", map_location=DEVICE))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

def cnn_manipulation_score(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img)
        probs = F.softmax(logits, dim=1)

    return float(probs[0][1])  # probability of "manipulated"
