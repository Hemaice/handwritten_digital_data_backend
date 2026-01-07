from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io

app = FastAPI()

# ------------------
# CORS Middleware
# ------------------
origins = [
    "*",  # For testing. Later replace with your frontend URL, e.g., "https://your-frontend.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------
# CNN Model (same as training)
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
        self.fc = nn.Linear(64 * 16 * 64, out_dim)

    def forward(self, x):
        return self.fc(self.cnn(x))

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
predictor.load_state_dict(
    torch.load("models/personality_predictor.pth", map_location="cpu"),
    strict=False
)
predictor.eval()

# ------------------
# Image transform
# ------------------
transform = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# ------------------
# PREDICT ENDPOINT
# ------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = transform(img).unsqueeze(0)

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

    result = dict(zip(traits, scores))
    result["dominant_trait"] = traits[scores.index(max(scores))]

    return result
