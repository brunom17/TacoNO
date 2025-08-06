from fastapi import FastAPI, UploadFile, File
from PIL import Image
import pytesseract
import io

app = FastAPI()

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        text = pytesseract.image_to_string(image)
        return {"text": text.strip()}
    except Exception as e:
        return {"error": str(e)}