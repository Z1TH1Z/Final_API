from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from typing import Dict
import torch
import io
import os
import webbrowser
import threading
from PIL import Image

# Local module imports
from ocr_proc import extract_meter_info
from inference_roof_type import RoofClassifierCNN, transform, CLASS_NAMES, DEVICE

# Ensure any required directories exist
os.makedirs("training", exist_ok=True)

# -------------------------------
# FastAPI App Setup
# -------------------------------
app = FastAPI(
    title="Unified ML API",
    description="Electric Meter OCR and Roof Type Classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# -------------------------------
# Redirect root to Swagger UI
# -------------------------------
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to SwaggerUI"""
    return RedirectResponse(url="/docs")

# -------------------------------
# Health Check
# -------------------------------
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "message": "API is running"}

# -------------------------------
# Load Roof Classifier Model
# -------------------------------
roof_model = RoofClassifierCNN().to(DEVICE)
roof_model.load_state_dict(torch.load("roof_type_cnn_best.pth", map_location=DEVICE))
roof_model.eval()

# -------------------------------
# Endpoint: Electric Meter OCR
# -------------------------------
@app.post("/ocr/meter", tags=["OCR"])
async def extract_ocr_data(file: UploadFile = File(...)) -> Dict:
    try:
        contents = await file.read()
        extracted_info = extract_meter_info(contents)
        return {"success": True, "data": extracted_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

# -------------------------------
# Endpoint: Roof Type Classifier
# -------------------------------
@app.post("/roof/classify", tags=["Roof Type Classifier"])
async def classify_roof(file: UploadFile = File(...)) -> Dict:
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = roof_model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
            confidence = torch.softmax(outputs, dim=1)[0][class_idx].item()

        return {
            "success": True,
            "predicted_roof_type": CLASS_NAMES[class_idx],
            "confidence": f"{confidence * 100:.2f}%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Roof classification failed: {str(e)}")

# -------------------------------
# Auto-open Swagger UI
# -------------------------------
def open_browser():
    webbrowser.open("http://127.0.0.1:8000/docs")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
