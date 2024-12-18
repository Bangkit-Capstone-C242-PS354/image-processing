import base64
import json
from flask import Flask, request, jsonify
from model import image_processing
from utils import download_from_gcs, save_to_firestore
import logging
from functools import lru_cache
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Set to store currently processing images
_processing = set()

def was_recently_processed(bucket: str, name: str) -> bool:
    """Check if an image is currently being processed."""
    key = f"{bucket}/{name}"
    
    if key in _processing:
        logger.info(f"Caught duplicate (within 1ms) for {key}")
        return True
    
    _processing.add(key)
    return False

@app.route("/", methods=["POST"])
def index():
    """Handle Cloud Storage notification via Pub/Sub."""
    try:
        envelope = request.get_json()
        if not envelope:
            msg = "no Pub/Sub message received"
            logger.error(f"error: {msg}")
            return jsonify({"error": msg, "success": False}), 200

        if not isinstance(envelope, dict) or "message" not in envelope:
            msg = "invalid Pub/Sub message format"
            logger.error(f"error: {msg}")
            return jsonify({"error": msg, "success": False}), 200

        pubsub_message = envelope["message"]
        
        if isinstance(pubsub_message, dict) and "data" in pubsub_message:
            data = base64.b64decode(pubsub_message["data"]).decode("utf-8")
            event = json.loads(data)
            
            # Skip processing for delete events
            if event.get('eventType') == 'OBJECT_DELETE':
                logger.info(f"Skipping deleted object: {event.get('name')}")
                return '', 204
            
            # Extract relevant information
            bucket = event['bucket']
            name = event['name']
            key = f"{bucket}/{name}"
            
            # If this is currently processing, skip it
            if was_recently_processed(bucket, name):
                return '', 204
            
            image_url = f"https://storage.googleapis.com/{bucket}/{name}"
            
            logger.info(f"Processing image from bucket: {bucket}, name: {name}")
            
            try:
                # Download image from GCS
                logger.info(f"Downloading image from GCS: {bucket}/{name}")
                image = download_from_gcs(bucket, name)
                
                # Process the image
                logger.info("Starting image processing")
                extracted_data = image_processing(image)
                logger.info("Image processing completed successfully")
                
                # Save to Firestore
                logger.info("Saving results to Firestore")
                doc_id = save_to_firestore(
                    image_url=image_url,
                    extracted_data=extracted_data,
                    bucket=bucket,
                    file_name=name
                )
                
                # Prepare response
                response_data = {
                    'success': True,
                    'document_id': doc_id,
                    'bucket': bucket,
                    'name': name,
                    'results': extracted_data,
                    'contentType': event.get('contentType'),
                    'timeCreated': event.get('timeCreated')
                }
                
                logger.info(f"Successfully processed and saved image: {name}")
                return jsonify(response_data), 200
                
            except Exception as e:
                error_msg = f"Error processing image: {str(e)}"
                logger.error(error_msg)
                # Return 200 instead of 500 to acknowledge the message
                return jsonify({
                    "error": error_msg,
                    "success": False,
                    "bucket": bucket,
                    "name": name
                }), 200
            finally:
                # Always remove from processing set when done
                _processing.discard(key)
        else:
            msg = "Invalid message format: missing data field"
            logger.error(f"error: {msg}")
            return jsonify({"error": msg, "success": False}), 200
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg, "success": False}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)