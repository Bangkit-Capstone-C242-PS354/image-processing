import base64
import json
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=["POST"])
def index():
    """Handle Cloud Storage notification via Pub/Sub."""
    envelope = request.get_json()
    if not envelope:
        msg = "no Pub/Sub message received"
        print(f"error: {msg}")
        return f"Bad Request: {msg}", 400

    if not isinstance(envelope, dict) or "message" not in envelope:
        msg = "invalid Pub/Sub message format"
        print(f"error: {msg}")
        return f"Bad Request: {msg}", 400

    pubsub_message = envelope["message"]
    
    if isinstance(pubsub_message, dict) and "data" in pubsub_message:
        data = base64.b64decode(pubsub_message["data"]).decode("utf-8")
        event = json.loads(data)
        
        # Extract relevant information
        bucket = event['bucket']
        name = event['name']
        image_url = f"gs://{bucket}/{name}"
        
        response_data = {
            'image_url': image_url,
            'bucket': bucket,
            'name': name,
            'contentType': event.get('contentType'),
            'size': event.get('size'),
            'timeCreated': event.get('timeCreated')
        }
        
        print(f"Processed notification: {response_data}")
        return response_data, 200

    return ("", 204)