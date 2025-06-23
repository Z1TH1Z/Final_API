import easyocr
import cv2
import numpy as np
import re

reader = easyocr.Reader(['en'])

def preprocess_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return thresh

def extract_meter_info(image_bytes):

    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    processed_image = preprocess_image(image)
    results = reader.readtext(processed_image)
    print("[OCR Results]")
    for bbox, text, conf in results:
        print(f"Text: {text}, Confidence: {conf:.2f}")


    extracted_info = {
        "kh": None,
        "frequency": None,
        "voltage": None,
        "serial_number": None,
        "other_specs": []
    }


    kh_pattern = re.compile(r"\bK\s*H\s*[:=]?\s*([0-9.]+)", re.IGNORECASE)
    freq_pattern = re.compile(r"\b([4-6]0)\s*(hz)?\b", re.IGNORECASE)
    volt_pattern = re.compile(r"\b([1-4][0-9]{2})\s*(v|volt|volts)?\b", re.IGNORECASE)
    serial_pattern = re.compile(r"\b(?:S/N|SN|Serial\s*(?:No|Number)?[:\s]*)?(\d{6,})\b", re.IGNORECASE)

    for (_, text, _) in results:
        text_clean = text.strip()

        if not extracted_info["kh"]:
            if kh_match := kh_pattern.search(text_clean):
                extracted_info["kh"] = kh_match.group(1)

        if not extracted_info["frequency"]:
            if freq_match := freq_pattern.search(text_clean):
                extracted_info["frequency"] = freq_match.group(1)

        if not extracted_info["voltage"]:
            if volt_match := volt_pattern.search(text_clean):
                extracted_info["voltage"] = volt_match.group(1)

        if not extracted_info["serial_number"]:
            if serial_match := serial_pattern.search(text_clean):
                if not re.search(r"hz|v|kh", text_clean.lower()):
                    extracted_info["serial_number"] = serial_match.group(1)

        extracted_info["other_specs"].append(text_clean)

    # Normalize output units
    if extracted_info["voltage"]:
        extracted_info["voltage"] += " V"
    if extracted_info["frequency"]:
        extracted_info["frequency"] += " Hz"
    if extracted_info["kh"]:
        extracted_info["kh"] += " Kh"

    return extracted_info
