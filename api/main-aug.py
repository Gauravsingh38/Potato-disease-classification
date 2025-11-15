                        #main-aug.py
                           
# ===============================================
# ✅ FASTAPI APP FOR POTATO DISEASE CLASSIFICATION
# Uses TensorFlow SavedModel (.pb format)
# ===============================================

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
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
# 🧠 LOAD MODEL (TensorFlow SavedModel .pb format)
# ==========================================================
model = tf.saved_model.load("../saved_models/my_aug_model_pb")

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
    """
    Convert uploaded image bytes into a normalized NumPy array.
    """
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))     # Resize to match model input
    image = np.array(image) / 255.0      # Normalize pixel values (0–1)
    return image

# ------------------------------------------------
# 🤖 Prediction endpoint
# ------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform prediction using the SavedModel signature
    infer = model.signatures["serving_default"]
    prediction = infer(tf.constant(img_batch, dtype=tf.float32))
    predictions = list(prediction.values())[0].numpy()

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    # Return response
    return {
        "class": predicted_class,
        "confidence": confidence
    }

# ------------------------------------------------
# ▶️ Run FastAPI app
# ------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)
