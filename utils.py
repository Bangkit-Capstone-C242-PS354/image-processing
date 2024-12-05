import requests
import numpy as np
from PIL import Image
from io import BytesIO
from google.cloud import storage
from google.cloud import firestore
from datetime import datetime

def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        raise Exception(f"Error downloading image from URL: {str(e)}")

def download_from_gcs(bucket_name: str, blob_name: str) -> Image.Image:
    """Download image from Google Cloud Storage."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Download as bytes
        image_bytes = blob.download_as_bytes()
        
        # Convert to PIL Image
        return Image.open(BytesIO(image_bytes))
    except Exception as e:
        raise Exception(f"Error downloading from GCS: {str(e)}")

def save_to_firestore(image_url: str, extracted_data: dict, bucket: str, file_name: str) -> str:
    """
    Save image URL and extracted data to Firestore
    Returns: Document ID of the created record
    """
    try:
        # Initialize Firestore client
        db = firestore.Client()
        
        # Create a new document in 'receipts' collection
        receipts_ref = db.collection('receipts')
        
        # Prepare data to save
        data = {
            'image_url': image_url,
            'bucket': bucket,
            'file_name': file_name,
            'extracted_data': extracted_data,
            'created_at': datetime.now(),
            'status': 'processed'
        }
        
        # Add document to Firestore
        doc_ref = receipts_ref.add(data)[1]
        
        return doc_ref.id
        
    except Exception as e:
        raise Exception(f"Error saving to Firestore: {str(e)}") 