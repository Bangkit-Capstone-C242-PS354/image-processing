import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Dict, List, Tuple
import easyocr
import re
import json
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning, message='Neither CUDA nor MPS are available.*')

class EasyReceiptOCR:
    def __init__(self, lang=['en', 'id']):
        # Use the pre-downloaded models from the container
        model_dir = '/app/models'
        
        # Initialize EasyOCR with specified languages and model directory
        self.reader = easyocr.Reader(
            lang,
            model_storage_directory=model_dir,
            download_enabled=False  # Disable downloading since we have the models
        )

    def preprocess_image2(self, image) -> np.ndarray:
        """
        Preprocessing menggunakan PIL.ImageEnhance:
        - Meningkatkan kontras
        - Menyesuaikan kecerahan
        - Menajamkan gambar
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        # Konversi ke mode RGB (pastikan formatnya sesuai)
        image = image.convert('RGB')

        # Tingkatkan kontras
        enhancer_contrast = ImageEnhance.Contrast(image)
        image = enhancer_contrast.enhance(2.0)  # Tingkat kontras (sesuaikan nilainya)

        # Tingkatkan kecerahan
        enhancer_brightness = ImageEnhance.Brightness(image)
        image = enhancer_brightness.enhance(1.2)  # Tingkat kecerahan (sesuaikan nilainya)

        # Tingkatkan ketajaman
        enhancer_sharpness = ImageEnhance.Sharpness(image)
        image = enhancer_sharpness.enhance(2.5)  # Tingkat ketajaman (sesuaikan nilainya)

        # Konversi kembali ke format OpenCV (array NumPy) jika diperlukan
        image_np = np.array(image)
        return image_np

    def preprocess_image(self, image) -> np.ndarray:
        """
        Memproses gambar untuk meningkatkan akurasi OCR
        """
        # Baca gambar
        image = image

        # Konversi ke grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            21, 
            10
        )
        
        # Tingkatkan kontras menggunakan CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(binary)
        
        # Sharpening menggunakan unsharp masking
        gaussian_blur = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian_blur, -0.5, 0)
        # Reduce noise
        denoised = cv2.fastNlMeansDenoising(sharpened)
        
        return denoised

    def extract_text_and_location(self, image) -> Tuple[List[Dict], np.ndarray]:
        """
        Mengekstrak teks dan lokasinya dari gambar struck menggunakan EasyOCR
        """
        
        # Proses gambar
        processed_image = self.preprocess_image2(image)
        
        # Buat salinan gambar untuk visualisasi
        visualization_image = image.copy()
        
        # Ekstrak data dengan EasyOCR
        receipt_data = []
        
        # Detect text using EasyOCR
        results = self.reader.readtext(processed_image)
        
        # Process each detected text
        for (bbox, text, prob) in results:
            if prob > 0.3:  # Filter low confidence detections
                # Get coordinates
                (top_left, top_right, bottom_right, bottom_left) = bbox
                x1, y1 = map(int, top_left)
                x2, y2 = map(int, top_right)
                x3, y3 = map(int, bottom_right)
                x4, y4 = map(int, bottom_left)
                
                # Draw rectangle and text on visualization image
                pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                cv2.polylines(visualization_image, [pts], True, (0, 255, 0), 2)
                cv2.putText(visualization_image, text, 
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (255, 0, 0), 2)
                
                # Create word data in required format
                word_data = {
                    "words": [{
                        "quad": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "x3": x3,
                            "y3": y3,
                            "x4": x4,
                            "y4": y4
                        },
                        "text": text,
                    }],
                }
                receipt_data.append(word_data)
        
        return results, receipt_data, visualization_image

# Fungsi menghitung posisi relatif
def calculate_relative_position(item, reference_item):
    dx = (item["quad"]["x1"] + item["quad"]["x3"]) / 2 - (reference_item["quad"]["x1"] + reference_item["quad"]["x3"]) / 2
    dy = (item["quad"]["y1"] + item["quad"]["y3"]) / 2 - (reference_item["quad"]["y1"] + reference_item["quad"]["y3"]) / 2
    return [dx, dy]

def predict_stage1_and_stage2(input_data, model_stage1, model_stage2, char_to_idx, label_encoder_stage1, label_encoder_stage2, image_size):
    # Preprocess input data
    features = []
    texts = []
    for item in input_data:
        quad = item['quad']
        feature = [
            quad["x1"] / image_size['width'], quad["y1"] / image_size['height'], 
            quad["x2"] / image_size['width'], quad["y2"] / image_size['height'], 
            quad["x3"] / image_size['width'], quad["y3"] / image_size['height'], 
            quad["x4"] / image_size['width'], quad["y4"] / image_size['height']
        ]
        features.append(feature)
        texts.append(item['text'])

    features = np.array(features)
    char_sequences = [[char_to_idx[char] for char in text if char in char_to_idx] for text in texts]
    char_padded = pad_sequences(char_sequences, maxlen=50)

    # Predict stage 1
    predictions_stage1 = np.argmax(model_stage1.predict([features, char_padded]), axis=1)
    predictions_stage1_labels = label_encoder_stage1.inverse_transform(predictions_stage1)

    # Prepare data for stage 2
    reference_positions = {"total_text": None, "tax_text": None,}  #"payment_cash": None, "payment_card": None, "payment_emoney": None}
    for i, item in enumerate(input_data):
        if predictions_stage1_labels[i] == "total_text":
            reference_positions["total_text"] = item
        elif predictions_stage1_labels[i] == "tax_text":
            reference_positions["tax_text"] = item

    features_stage2 = []
    for item in input_data:
        quad = item['quad']
        feature = [
            quad["x1"] / image_size['width'], quad["y1"] / image_size['height'], 
            quad["x2"] / image_size['width'], quad["y2"] / image_size['height'], 
            quad["x3"] / image_size['width'], quad["y3"] / image_size['height'], 
            quad["x4"] / image_size['width'], quad["y4"] / image_size['height']
        ]
        if reference_positions["total_text"]:
            feature.extend(calculate_relative_position(item, reference_positions["total_text"]))
        else:
            feature.extend([0, 0])
        if reference_positions["tax_text"]:
            feature.extend(calculate_relative_position(item, reference_positions["tax_text"]))
        else:
            feature.extend([0, 0])
        features_stage2.append(feature)

    features_stage2 = np.array(features_stage2)

    # Predict stage 2
    predictions_stage2 = np.argmax(model_stage2.predict([features_stage2, char_padded]), axis=1)
    predictions_stage2_labels = label_encoder_stage2.inverse_transform(predictions_stage2)

    # Combine results
    results = []
    for i, item in enumerate(input_data):
        results.append({
            "text": item["text"],
            "quad": item["quad"],
            "stage1_label": predictions_stage1_labels[i],
            "stage2_label": predictions_stage2_labels[i],
        })

    return results

def extract_decimal_number(text):
    # Hapus simbol mata uang dan karakter non-angka lainnya
    cleaned_text = re.sub(r'[^\d,.\s]', '', text)  # Menghapus simbol selain angka, koma, titik, atau spasi
    cleaned_text = re.sub(r'\s*\.\s*', '', cleaned_text)  # Menghapus titik yang muncul setelah simbol mata uang
    # Ganti koma dengan titik untuk konsistensi desimal
    normalized = cleaned_text.replace(",", ".")
    # Hilangkan pemisah ribuan (titik di tengah angka)
    cleaned_number = re.sub(r'\.(?=\d{3}($|\D))', '', normalized)
    # Konversi ke float
    try:
        return float(cleaned_number)
    except ValueError:
        return 0  # Jika angka tidak valid
    
def extract_text_values(predicted_results):
    total_value = 0
    tax_value = 0

    payment_method = 'cash'

    for result in predicted_results:
        if result['stage2_label'] == "total_value":
            total_value = extract_decimal_number(result['text'])

        elif result['stage2_label'] == "tax_value":
            tax_value = extract_decimal_number(result['text'])

        if result['stage2_label'] == "payment_card":
            try:
                payment_method = 'card'
            except ValueError:
                payment_method = ''
        elif result['stage2_label'] == "payment_emoney":
            try:
                payment_method = 'emoney'
            except ValueError:
                payment_method = ''
        elif result['stage2_label'] == "payment_cash":
            try:
                payment_method = 'cash'
            except ValueError:
                payment_method = ''

    return {
        "total_value": total_value,
        "tax_value": tax_value,
        "payment_method": payment_method
    }

def clean_data(data):
    cleaned_data = []
    for item in data:
        for word in item.get('words', []):
            cleaned_data.append({
                'quad': word.get('quad'),
                'text': word.get('text')
            })
    return cleaned_data

def image_processing(image: Image.Image):
    """
    Process receipt image and extract relevant information.
    
    Args:
        image: PIL Image object
    Returns:
        dict: Extracted receipt data including total_value, tax_value, and payment_method
    """
    ocr = EasyReceiptOCR(['en', 'id'])
    
    # Convert PIL Image to NumPy array
    image_data = np.array(image)

    # Extract text and location
    easy_results, receipt_data, visualization_image = ocr.extract_text_and_location(image_data)

    # Get image dimensions for normalization
    if len(image_data.shape) == 3:
        image_height, image_width, _ = image_data.shape
    else:
        image_height, image_width = image_data.shape
    image_size = {'width': image_width, 'height': image_height}

    # Clean and process the extracted data
    cleaned_result = clean_data(receipt_data)

    # Load models and encoders from the models directory
    model_stage1 = load_model("models/model_stage1.h5")
    model_stage2 = load_model("models/model_stage2.h5")
    
    with open("models/char_to_idx.pkl", "rb") as file:
        char_to_idx = pickle.load(file)
    with open("models/label_encoder_stage1.pkl", "rb") as file:
        label_encoder_stage1 = pickle.load(file)
    with open("models/label_encoder_stage2.pkl", "rb") as file:
        label_encoder_stage2 = pickle.load(file)

    # Make predictions
    predicted_results = predict_stage1_and_stage2(
        input_data=cleaned_result,
        model_stage1=model_stage1,
        model_stage2=model_stage2,
        char_to_idx=char_to_idx,
        label_encoder_stage1=label_encoder_stage1,
        label_encoder_stage2=label_encoder_stage2,
        image_size=image_size
    )

    # Extract and return final results
    return extract_text_values(predicted_results)

# def main():
#     image_path = "contoh-image.jpg"
#     image_processing(image_path)

# if __name__ == "__main__":
#     main()