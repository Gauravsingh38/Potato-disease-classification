# ================================================
# 🚀 Potato Disease Classification using FastAPI + TensorFlow Serving
# ================================================

# Import required libraries
from fastapi import FastAPI, File, UploadFile           # For building REST API and file upload
from fastapi.middleware.cors import CORSMiddleware      # To allow cross-origin requests (frontend-backend connection)
import uvicorn                                          # For running FastAPI server
import numpy as np                                      # For numerical operations
from io import BytesIO                                  # For reading image bytes
from PIL import Image                                   # For image processing
import requests                                         # For sending HTTP requests to TensorFlow Serving

# Initialize FastAPI app with a title
app = FastAPI(title="Potato Disease Classification API (TF Serving)")

# ==================================================
# 🌐 Configure CORS (Cross-Origin Resource Sharing)
# ==================================================
# Allow requests from your frontend (like React app running on localhost:3000)
origins = ["http://localhost", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # Allowed origins
    allow_credentials=True,
    allow_methods=["*"],            # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],            # Allow all headers
)

# ==================================================
# 🔗 TensorFlow Serving endpoint (Docker container)
# ==================================================
# This URL corresponds to the TF Serving container running your model
ENDPOINT = "http://localhost:8501/v1/models/potatoes_model:predict"

# ==================================================
# 🏷️ Class labels corresponding to your model output
# ==================================================
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# ==================================================
# ✅ Health check route — confirms that API is alive
# ==================================================
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive!"}

# ==================================================
# 🖼️ Function to preprocess uploaded image
# ==================================================
def read_file_as_image(data) -> np.ndarray:
    # Read the image from uploaded bytes and convert to RGB format
    image = Image.open(BytesIO(data)).convert("RGB")
    # Resize to model input size
    image = image.resize((256, 256))
    # Convert image to numpy array and normalize (0-1 range)
    image = np.array(image) / 255.0
    return image

# ==================================================
# 🔮 Prediction route — sends image to TF Serving and returns result
# ==================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Step 1️⃣: Read and preprocess image
    image = read_file_as_image(await file.read())
    # Add batch dimension — model expects input shape (1, 256, 256, 3)
    img_batch = np.expand_dims(image, axis=0)

    # Step 2️⃣: Prepare JSON data and send request to TF Serving
    json_data = {"instances": img_batch.tolist()}
    response = requests.post(ENDPOINT, json=json_data)

    # Step 3️⃣: Handle TensorFlow Serving errors (if any)
    if response.status_code != 200:
        return {"error": f"TensorFlow Serving error: {response.text}"}

    # Step 4️⃣: Extract predictions from TF Serving response
    predictions = np.array(response.json()["predictions"][0])
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    # Step 5️⃣: Return final result to frontend / user
    return {"class": predicted_class, "confidence": confidence}

# ==================================================
# ▶️ Run the FastAPI server (localhost:8000)
# ==================================================
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
