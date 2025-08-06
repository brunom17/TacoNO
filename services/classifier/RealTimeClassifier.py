from ExtractBet import extract_text_from_image
import uuid
import os
import asyncio
import joblib
import pandas as pd
from telethon import TelegramClient, events
import re
import emoji
import csv
from datetime import datetime



# === CONFIG ===
api_id = 27991452
api_hash = 'b7360f600f8048135753611fe7edc6e3'
group_username = 'TestingTelethonBets'  # 👈 Replace with your actual group username or ID

# === Load Trained Model ===
model = joblib.load("message_classifier_pipeline.pkl")

# === Save the Label added by the model in csv file ===
def save_to_csv(text, label, confidence, emoji_count, media_present):
    file_path = "classified_messages.csv"
    file_exists = os.path.isfile(file_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(file_path, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "text", "confidence", "emoji_count", "media_present", "label"])
        writer.writerow([timestamp, text, confidence, emoji_count, media_present, label])

# === Text Cleaning and Emoji Counting ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+", "", text)  # remove URLs
    text = re.sub(r"[\\n\\r]+", " ", text)  # remove line breaks
    return text.strip()

def count_emojis(text):
    return len([ch for ch in text if ch in emoji.EMOJI_DATA])

# === Set up Telethon Client ===
client = TelegramClient("live_session", api_id, api_hash)

@client.on(events.NewMessage(chats=group_username))
async def handler(event):
    msg = event.message
    raw_text = msg.message or ""
    media_present = 1 if msg.media else 0

    # Preprocessing
    cleaned_text = clean_text(raw_text)
    emoji_count = count_emojis(raw_text)

    # Create DataFrame for prediction
    df = pd.DataFrame([{
        "text_clean": cleaned_text,
        "has_media": media_present,
        "emoji_count": emoji_count
    }])

    # Predict
    predicted_label = model.predict(df)[0]
    confidence = max(model.predict_proba(df)[0])

    # Print results
    print(f"\n📥 New message: {raw_text[:100]}...")
    print(f"📌 Predicted Label: {predicted_label} | 🔢 Confidence: {confidence:.2f}")
    print("------------------------------------------------------------")

    # Save the results in csv file
    save_to_csv(raw_text, predicted_label, confidence, emoji_count, media_present)

    # If 'bet_image', send image to your own account
    if predicted_label == "bet_image" and msg.media:
        try:
            my_user_id = 6407926288  # Replace with your actual ID

            # Create folder if it doesn't exist
            os.makedirs("images", exist_ok=True)

            # Generate a unique filename
            image_filename = f"{uuid.uuid4()}.jpg"
            image_path = os.path.join("images", image_filename)

            # Download image to local folder
            await msg.download_media(file=image_path)

            # Run OCR to extract text
            extracted_text = extract_text_from_image(image_path)

            # Send extracted text first
            await client.send_message(
                entity=my_user_id,
                message=f"🧾 *Extracted text from bet image:*\n\n```{extracted_text}```",
                parse_mode='markdown'
            )

            # Send image file with classification info
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

# === Start Client ===
with client:
    print("✅ Listening for new messages in real time...")
    client.run_until_disconnected()