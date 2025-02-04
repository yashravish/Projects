# ocr-parser/ocr_parser.py
import pytesseract
from PIL import Image
import sys

def perform_ocr(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        print("OCR Output:")
        print(text)
        return text
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ocr_parser.py <image_path>")
        sys.exit(1)
    perform_ocr(sys.argv[1])
