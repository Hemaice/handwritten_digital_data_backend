from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import io

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------
# CNN Feature Extractor
# ------------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc = nn.Linear(64 * 16 * 16, out_dim)   # FIXED DIMENSION

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)

# ------------------
# Personality Predictor
# ------------------
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


# ------------------
# Load models
# ------------------
cnn = CNNFeatureExtractor()
cnn.load_state_dict(torch.load("models/cnn_model.pth", map_location="cpu"))
cnn.eval()

predictor = PersonalityPredictor()
predictor.load_state_dict(torch.load("models/personality_predictor.pth", map_location="cpu"))
predictor.eval()

# ------------------
# FIXED TRANSFORM (Better for handwriting)
# ------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),     # Square, avoids distortion
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # Helps model stability
])

# ------------------
# FINAL FIXED PREDICT ENDPOINT
# ------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Read uploaded bytes
    image_bytes = await image.read()

    # Read image with Pillow
    img = Image.open(io.BytesIO(image_bytes))

    # ---- FIX transparency (canvas PNG) ----
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background.convert("RGB")

    img = img.convert("L")

    # Apply transform
    img = transform(img).unsqueeze(0)

    # Model Prediction
    with torch.no_grad():
        features = cnn(img)
        output = predictor(features)
        scores = torch.sigmoid(output).squeeze().tolist()

    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

    # Map result
    result = {trait: float(score) for trait, score in zip(traits, scores)}

    # Find dominant
    result["dominant_trait"] = traits[int(np.argmax(scores))]

    return result
