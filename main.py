import io
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import gdown
import keras
import numpy as np
from PIL import Image

app = FastAPI()

# Enable CORS so any front-end can talk to your backend safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "best_model.keras"

# Replace this with the File ID you copied from your Google Drive link!
GOOGLE_DRIVE_FILE_ID = "1UJ87K2Myv-Y_b0B8VzG7UAv04XSSD0f_"


@app.on_event("startup")
def load_model():
    global model
    # If the model isn't downloaded yet, pull it from Google Drive automatically
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    print("Loading Keras model...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image bytes
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content)).convert("RGB")

        # Your exact model requirement: 128x128 pixels
        img = img.resize((128, 128))

        # Normalize data values
        img_array = np.array(img, dtype=np.float32) / 255.0
        input_data = np.expand_dims(img_array, axis=0)

        # Run inference
        predictions = model.predict(input_data)
        score = float(predictions[0][0])

        # Map binary classifications
        # Assumes closer to 1 means Fake, closer to 0 means Real (Swap if yours is inverted)
        real_prob = 1.0 - score
        fake_prob = score

        return {
            "status": "success",
            "predictions": {
                "Authentic (Real)": round(real_prob, 4),
                "Manipulated (Fake)": round(fake_prob, 4),
            },
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/")
def read_root():
    return {"message": "Real vs Fake Detector API by Taj is running online!"}
