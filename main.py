from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import uvicorn
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "fruit_model.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} was not found. Put it in the same folder as main.py")

model = tf.keras.models.load_model(MODEL_PATH)

IMG_HEIGHT = 180
IMG_WIDTH = 180

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def home():
    return {"message": "Fruit classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)
    prediction = model.predict(img_array, verbose=0)
    raw_value = float(prediction[0][0])

    if raw_value >= 0.5:
        predicted_label = "orange"
        orange_percentage = raw_value * 100
        apple_percentage = (1 - raw_value) * 100
    else:
        predicted_label = "apple"
        apple_percentage = (1 - raw_value) * 100
        orange_percentage = raw_value * 100

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "filename": file.filename,
        "predicted_label": predicted_label,
        "apple_percentage": round(apple_percentage, 2),
        "orange_percentage": round(orange_percentage, 2),
        "raw_model_output": round(raw_value, 4),
        "image_base64": image_base64,
        "image_mime_type": file.content_type
    }

port = int(os.environ.get("PORT", 8080))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port)