from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pathlib import Path

app = FastAPI()

# Allow mobile/web clients during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup with a robust path
# The model is in H5 format (legacy)
MODEL_PATH = Path(__file__).resolve().parent / "potato_disease_model.h5"

try:
    # Load as H5 format (legacy)
    MODEL = tf.keras.models.load_model(str(MODEL_PATH))
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, API is working"}

# convert file → numpy image
def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    image = image.resize((256, 256))  # IMPORTANT: match training size used in training
    image = np.asarray(image, dtype=np.float32) / 255.0  # normalize if you trained like this
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = read_file_as_image(contents)
        img_batch = np.expand_dims(image, axis=0)

        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[int(np.argmax(predictions[0]))]
        confidence = float(np.max(predictions[0]))

        return {"class": predicted_class, "confidence": confidence}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)