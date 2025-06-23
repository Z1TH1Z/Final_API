from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from typing import Dict
import torch
import io
import os
from PIL import Image

# Local imports
from ocr_proc import extract_meter_info
from inference_roof_type import RoofClassifierCNN, transform, CLASS_NAMES, DEVICE

# FastAPI app
app = FastAPI(
    title="Electric Meter + Roof Classifier API",
    description="Electric Meter OCR and Roof Type Classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Redirect root to Swagger UI
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "message": "API is running"}

# Load the roof model once (on cold start)
roof_model = RoofClassifierCNN().to(DEVICE)
roof_model.load_state_dict(torch.load("roof_type_cnn_best.pth", map_location=DEVICE))
roof_model.eval()

@app.post("/ocr/meter", tags=["OCR"])
async def extract_ocr_data(file: UploadFile = File(...)) -> Dict:
    try:
        contents = await file.read()
        extracted_info = extract_meter_info(contents)
        return {"success": True, "data": extracted_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

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

# ✅ DO NOT add `if __name__ == "__main__"` block here.
# Render will run `uvicorn main:app --host 0.0.0.0 --port 10000`
