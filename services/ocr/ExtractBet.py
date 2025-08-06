from PIL import Image
import pytesseract
import os

# Optional (Windows only): Explicit Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"[ERROR] OCR failed: {e}"

# Optional local test
if __name__ == "__main__":
    text = extract_text_from_image("sample.jpg")
    print(text)