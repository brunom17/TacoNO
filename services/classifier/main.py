from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import pandas as pd
import joblib
import requests
import re
import emoji

app = FastAPI()

model = joblib.load("message_classifier_pipeline.pkl")

OCR_SERVICE_URL = "http://ocr:5000/extract"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[\\n\\r]+", " ", text)
    return text.strip()

def count_emojis(text):
    return len([ch for ch in text if ch in emoji.EMOJI_DATA])

@app.post("/classify")
async def classify_message(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    cleaned_text = clean_text(message)
    emoji_count = count_emojis(message)
    has_media = 1 if file else 0

    df = pd.DataFrame([{
        "text_clean": cleaned_text,
        "has_media": has_media,
        "emoji_count": emoji_count
    }])

    predicted_label = model.predict(df)[0]
    confidence = max(model.predict_proba(df)[0])

    ocr_text = None
    if predicted_label == "bet_image" and file:
        response = requests.post(OCR_SERVICE_URL, files={"file": (file.filename, await file.read())})
        ocr_text = response.json().get("text")

    return {
        "label": predicted_label,
        "confidence": confidence,
        "ocr_text": ocr_text
    }