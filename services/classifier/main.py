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
from telethon import TelegramClient, events
from datetime import datetime

app = FastAPI()

# === Load Trained Model ===
model = joblib.load("message_classifier_pipeline.pkl")

# === OCR Microservice URL ===
OCR_SERVICE_URL = "http://ocr:5000/extract"

# === Telegram Setup ===
api_id = 27991452
api_hash = 'b7360f600f8048135753611fe7edc6e3'
group_username = 'TestingTelethonBets'  # 👈 Replace with your group
my_user_id = 6407926288  # 👈 Replace with your own ID
client = TelegramClient("live_session", api_id, api_hash)

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

# === Real-Time Telegram Handler ===
@client.on(events.NewMessage(chats=group_username))
async def telegram_handler(event):
    msg = event.message
    raw_text = msg.message or ""
    media_present = 1 if msg.media else 0

    cleaned_text = clean_text(raw_text)
    emoji_count = count_emojis(raw_text)

    df = pd.DataFrame([{
        "text_clean": cleaned_text,
        "has_media": media_present,
        "emoji_count": emoji_count
    }])

    predicted_label = model.predict(df)[0]
    confidence = max(model.predict_proba(df)[0])

    save_to_csv(raw_text, predicted_label, confidence, emoji_count, media_present)

    print(f"\n📥 New message: {raw_text[:100]}...")
    print(f"📌 Predicted Label: {predicted_label} | 🔢 Confidence: {confidence:.2f}")

    if predicted_label == "bet_image" and msg.media:
        try:
            os.makedirs("images", exist_ok=True)
            image_filename = f"{uuid.uuid4()}.jpg"
            image_path = os.path.join("images", image_filename)
            await msg.download_media(file=image_path)

            with open(image_path, "rb") as img:
                response = requests.post(OCR_SERVICE_URL, files={"file": (image_filename, img)})
                extracted_text = response.json().get("text")

            await client.send_message(
                entity=my_user_id,
                message=f"🧾 *Extracted text from bet image:*\n\n```{extracted_text}```",
                parse_mode='markdown'
            )

            await client.send_file(
                entity=my_user_id,
                file=image_path,
                caption=(
                    f"🖼️ *Image classified as:* `{predicted_label}`\n"
                    f"📊 *Confidence:* `{confidence:.2f}`"
                ),
                parse_mode='markdown'
            )

            print("\n🧾 OCR Extracted Text:")
            print(extracted_text)

        except Exception as e:
            print(f"⚠️ Error processing bet_image message: {e}")

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
        response = requests.post(OCR_SERVICE_URL, files={"file": (file.filename, await file.read())})
        ocr_text = response.json().get("text")

    return {
        "label": predicted_label,
        "confidence": confidence,
        "ocr_text": ocr_text
    }

# === Start Telegram Listener ===
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(client.start())
    print("🚀 Telegram client started in background.")
    await asyncio.sleep(3)
    asyncio.create_task(client.run_until_disconnected())