from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
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
CLASS_NAMES_PATH = "class_names.txt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} was not found. Put it in the same folder as main.py")

# `compile=False` avoids deserializing training-only compile state, which can
# break when the model is served with a different Keras/TensorFlow stack.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

IMG_HEIGHT = 180
IMG_WIDTH = 180

if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        CLASS_NAMES = [line.strip() for line in file if line.strip()]
else:
    CLASS_NAMES = ["apple", "orange"]

def preprocess_image(image_bytes):
    image = tf.keras.utils.load_img(
        io.BytesIO(image_bytes),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
    )
    img_array = tf.keras.utils.img_to_array(image).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def home():
    return {"message": "Fruit classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)
    prediction = model.predict(img_array, verbose=0)[0]
    probabilities = tf.nn.softmax(prediction).numpy()
    predicted_index = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_index]

    scores = {
        class_name: round(float(probability * 100), 2)
        for class_name, probability in zip(CLASS_NAMES, probabilities)
    }

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "filename": file.filename,
        "predicted_label": predicted_label,
        "class_names": CLASS_NAMES,
        "scores": scores,
        "raw_model_output": [round(float(value), 4) for value in prediction],
        "image_base64": image_base64,
        "image_mime_type": file.content_type
    }

port = int(os.environ.get("PORT", 8080))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port)
