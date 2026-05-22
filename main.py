import io
import os
import threading
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import gdown
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.tflite"
# PASTE YOUR NEW TFLITE GOOGLE DRIVE FILE ID HERE!
GOOGLE_DRIVE_FILE_ID = "1UJ87K2Myv-Y_b0B8VzG7UAv04XSSD0f_"

interpreter = None
input_details = None
output_details = None
model_loading_status = "Not started"


def background_load_model():
    global interpreter, input_details, output_details, model_loading_status
    try:
        if not os.path.exists(MODEL_PATH):
            model_loading_status = "Downloading TFLite model..."
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

        model_loading_status = "Loading TFLite Interpreter..."
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        model_loading_status = "Ready"
        print("⚡ Lightweight TFLite Model loaded successfully!")
    except Exception as e:
        model_loading_status = f"Failed to load: {str(e)}"
        print(f"Error: {e}")


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=background_load_model)
    thread.start()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global interpreter, input_details, output_details, model_loading_status

    if interpreter is None:
        return {
            "status": "error",
            "message": f"Model not ready. Status: {model_loading_status}",
        }

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img = img.resize((128, 128))

        img_array = np.array(img, dtype=np.float32) / 255.0
        input_data = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])

        score = float(predictions[0][0])
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
    return {
        "message": "Lightweight Detector API is online!",
        "model_status": model_loading_status,
    }
