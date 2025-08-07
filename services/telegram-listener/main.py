from telethon import TelegramClient, events
import requests
import re
import emoji
from datetime import datetime
import os
import tempfile

api_id = 27991452
api_hash = 'b7360f600f8048135753611fe7edc6e3'
group_username = 'TestingTelethonBets'  # 👈 Replace if needed
client = TelegramClient("live_session", api_id, api_hash)

CLASSIFIER_API_URL = "http://localhost:5001/classify"  # Endpoint exposed by classifier microservice

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[\\n\\r]+", " ", text)
    return text.strip()

def count_emojis(text):
    return len([ch for ch in text if ch in emoji.EMOJI_DATA])

import tempfile

@client.on(events.NewMessage(chats=group_username))
async def handler(event):
    print("[📥] New message received from group!")
    
    text = event.message.message
    cleaned = clean_text(text)
    emoji_count = count_emojis(cleaned)

    files = None
    file_bytes = None

    if event.message.media:
        print("[📸] Downloading media...")
        file_bytes = await event.message.download_media(bytes)
        files = {
            "file": ("image.jpg", file_bytes)
        }

    try:
        response = requests.post(
            CLASSIFIER_API_URL,
            data={"message": cleaned},
            files=files
        )

        print("[🛰️] Raw response text:", response.text)
        result = response.json()
        label = result.get("label")
        confidence = result.get("confidence")
        ocr_text = result.get("ocr_text")

        reply_text = f"""
🧠 *Prediction Result*
--------------------------
📩 *Message:* {text}
🏷️ *Label:* `{label}`
📊 *Confidence:* `{confidence:.2f}`
📝 *OCR Text:* `{ocr_text or 'N/A'}`
""".strip()

        if file_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name

            await event.reply(reply_text, file=temp_file_path)
            print("[✅] Replied with image and prediction")
        else:
            await event.reply(reply_text)
            print("[✅] Replied with prediction (text only)")

    except Exception as e:
        print(f"[❌] Error in handler: {e}")

print("[⏳] Starting Telegram client...")
try:
    client.start()
    print("[🚀] Telegram listener started successfully!")
    client.run_until_disconnected()
except Exception as e:
    print(f"[❌] Error during Telegram client startup: {e}")

