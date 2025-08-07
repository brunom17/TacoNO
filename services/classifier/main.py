from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import pandas as pd
import joblib
import requests
import re
import emoji
import asyncio
import os
import csv
import uuid
from datetime import datetime

app = FastAPI()

# === Load Trained Model ===
model = joblib.load("message_classifier_pipeline.pkl")

# === OCR Microservice URL ===
OCR_SERVICE_URL = "http://localhost:5002/extract"

# === Utils ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[\\n\\r]+", " ", text)
    return text.strip()

def count_emojis(text):
    return len([ch for ch in text if ch in emoji.EMOJI_DATA])

def save_to_csv(text, label, confidence, emoji_count, media_present):
    file_path = "classified_messages.csv"
    file_exists = os.path.isfile(file_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(file_path, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "text", "confidence", "emoji_count", "media_present", "label"])
        writer.writerow([timestamp, text, confidence, emoji_count, media_present, label])

# === API Endpoint ===
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
        file_content = await file.read()  # Read it once and reuse
        try:
            response = requests.post(
                OCR_SERVICE_URL,
                files={"file": (file.filename, file_content)}
            )
            response.raise_for_status()
            ocr_text = response.json().get("text")
        except Exception as e:
            print(f"[❌] OCR error: {e}")
            ocr_text = None

    return {
        "label": predicted_label,
        "confidence": confidence,
        "ocr_text": ocr_text
    }
