from flask import Flask, request, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model Definition
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNNModel()

# Use current working directory instead of __file__
MODEL_PATH = os.path.join(os.getcwd(), "cnn_mnist_model.pth")

# Ensure the model is loaded from the correct path
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None
    error = None

    if request.method == "POST":
        try:
            if "file" not in request.files:
                error = "No file uploaded"
                return render_template("index.html", error=error)

            file = request.files["file"]

            if file.filename == "":
                error = "No selected file"
                return render_template("index.html", error=error)

            # Save file
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            print(f"Uploaded file saved to: {filepath}")

            # Open and preprocess image
            try:
                image = Image.open(filepath).convert("L")
                img = transform(image).unsqueeze(0)
            except Exception as e:
                error = f"Error loading image: {str(e)}"
                return render_template("index.html", error=error)

            print(f"Image tensor shape: {img.shape}")

            # Predict
            with torch.no_grad():
                output = model(img)
                _, pred = torch.max(output, 1)
                prediction = int(pred.item())

            image_path = filepath

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path,
        error=error
    )


# Local Run (ignored by Render)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
