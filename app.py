from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import io

app = FastAPI()

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ CNN Feature Extractor (MATCHES TRAINED MODEL) ------------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   # 128x128 → 128x128
            nn.ReLU(),
            nn.MaxPool2d(2),                  # → 64x64

            nn.Conv2d(32, 64, 3, padding=1),  # 64x64 → 64x64
            nn.ReLU(),
            nn.MaxPool2d(2),                  # → 32x32

            nn.Flatten()
        )

        # MUST MATCH TRAINING: 64 * 32 * 32 = 65536
        self.fc = nn.Linear(64 * 32 * 32, out_dim)

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)

# ------------------ Personality Predictor ------------------
class PersonalityPredictor(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.net(x)

# ------------------ Load Models ------------------
cnn = CNNFeatureExtractor()
cnn.load_state_dict(torch.load("models/cnn_model.pth", map_location="cpu"))
cnn.eval()

predictor = PersonalityPredictor()
predictor.load_state_dict(torch.load("models/personality_predictor.pth", map_location="cpu"))
predictor.eval()

# ------------------ FIXED Image Transform ------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # EXACT SIZE NEEDED
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ------------------ Prediction Endpoint ------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    # Read file bytes
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes))

    # FIX: Transparent PNG from canvas → convert to white background
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background.convert("L")
    else:
        img = img.convert("L")

    # Transform to tensor
    img = transform(img).unsqueeze(0)

    # Run through models
    with torch.no_grad():
        features = cnn(img)
        output = predictor(features)
        scores = torch.sigmoid(output).squeeze().tolist()

    traits = [
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism"
    ]

    # Format result
    result = {trait: float(score) for trait, score in zip(traits, scores)}
    result["dominant_trait"] = traits[int(np.argmax(scores))]

    return result
