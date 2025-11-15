# ===============================================
# ✅ FASTAPI APP FOR POTATO DISEASE CLASSIFICATION
# Supports two model loading methods:
#   1️⃣ Keras Model (Architecture + Weights)  ← Default Active
#   2️⃣ TensorFlow SavedModel (.pb format)     ← Alternate Option
# ===============================================

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from keras.models import load_model
from keras import layers, models
import tensorflow as tf

# ------------------------------------------------
# 🚀 Initialize FastAPI app
# ------------------------------------------------
app = FastAPI()

# ------------------------------------------------
# 🌐 Allow CORS (for frontend access, e.g. React)
# ------------------------------------------------
origins = ["http://localhost", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# 🧩 1️⃣ LOAD MODEL (Keras Architecture + Weights)  ← ACTIVE
# ==========================================================

USE_KERAS_MODEL = True  # 🔁 Toggle between Keras (True) and SavedModel (False)

if USE_KERAS_MODEL:
    print("🔹 Loading Keras model (architecture + weights)...")

    resize_and_rescale = layers.Rescaling(1. / 255)
    input_shape = (256, 256, 3)
    n_classes = 3

    model = models.Sequential([
        resize_and_rescale,
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])

    # Load the trained weights only
    model.load_weights("../saved_models/my_model/weights.weights.h5")

else:
    # ==========================================================
    # 🧠 2️⃣ LOAD MODEL (TensorFlow SavedModel .pb format)
    # ==========================================================
    print("🔹 Loading TensorFlow SavedModel (.pb format)...")

    model = tf.saved_model.load("../saved_models/my_model_pb")
    infer = model.signatures["serving_default"]  # ✅ Serving signature for inference

# ------------------------------------------------
# 🏷️ Define class labels
# ------------------------------------------------
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# ------------------------------------------------
# 🔍 Health check route
# ------------------------------------------------
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

# ------------------------------------------------
# 🖼️ Helper function — convert uploaded image to NumPy array
# ------------------------------------------------
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return image

# ------------------------------------------------
# 🤖 Prediction endpoint
# ------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    if USE_KERAS_MODEL:
        predictions = model.predict(img_batch)
    else:
        output = infer(tf.constant(img_batch, dtype=tf.float32))
        output_key = list(output.keys())[0]
        predictions = output[output_key].numpy()

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        "class": predicted_class,
        "confidence": confidence
    }

# ------------------------------------------------
# ▶️ Run FastAPI app
# ------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)