import base64
import json
from flask import Flask, request, jsonify
from model import image_processing
from utils import download_from_gcs, save_to_firestore
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/", methods=["POST"])
def index():
    """Handle Cloud Storage notification via Pub/Sub."""
    try:
        envelope = request.get_json()
        if not envelope:
            msg = "no Pub/Sub message received"
            logger.error(f"error: {msg}")
            return jsonify({"error": msg}), 400

        if not isinstance(envelope, dict) or "message" not in envelope:
            msg = "invalid Pub/Sub message format"
            logger.error(f"error: {msg}")
            return jsonify({"error": msg}), 400

        pubsub_message = envelope["message"]
        
        if isinstance(pubsub_message, dict) and "data" in pubsub_message:
            data = base64.b64decode(pubsub_message["data"]).decode("utf-8")
            event = json.loads(data)
            
            # Extract relevant information
            bucket = event['bucket']
            name = event['name']
            image_url = f"https://storage.googleapis.com/{bucket}/{name}"
            
            logger.info(f"Processing image from bucket: {bucket}, name: {name}")
            
            try:
                # Download image from GCS
                image = download_from_gcs(bucket, name)
                
                # Process the image
                extracted_data = image_processing(image)
                
                # Save to Firestore
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
                return jsonify({"error": error_msg}), 500
    
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)