from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, everyone! This is the skin lesion detection app!"}

# Load the TFLite model
MODEL_PATH = "model.tflite"
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    raise RuntimeError(f"Failed to load the TFLite model: {e}")

# Define the response model
class PredictionResponse(BaseModel):
    lesion_type: str
    confidence: float

# Define possible lesion types (update based on your model)
LESION_TYPES = ["Melanoma", "Basal Cell Carcinoma", "Actinic Keratosis", "Benign Keratosis", 
                "Dermatofibroma", "Vascular Lesion", "Nevus"]

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess image (resize, normalize)
        image = image.resize((224, 224))  # Match your model's input size
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0).astype(np.float32)
        
        # Run the TFLite model
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Extract the top prediction
        predicted_index = np.argmax(output_data)
        confidence = float(np.max(output_data))
        lesion_type = LESION_TYPES[predicted_index]
        
        # Return JSON response
        return PredictionResponse(lesion_type=lesion_type, confidence=confidence)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
