#!/usr/bin/env python3
"""
Railway.app Whisper Proxy API
Ultra-minimal: Uses Hugging Face API (no local model)
Perfect for Railway's free tier
"""

import os
import logging
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
HF_TOKEN = os.environ.get('HF_TOKEN')  # Set this in Railway environment

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'm4a', 'flac', 'ogg', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation"""
    return jsonify({
        'service': 'Free Whisper Speech-to-Text API',
        'status': 'online',
        'type': 'proxy',
        'backend': 'Hugging Face',
        'model': 'openai/whisper-small',
        'platform': 'Railway.app',
        'cost': 'FREE',
        'endpoints': {
            'health': 'GET /',
            'transcribe': 'POST /transcribe (multipart/form-data with "audio" field)',
            'models': 'GET /models'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'note': 'Requires HF_TOKEN environment variable'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'type': 'proxy',
        'backend': 'huggingface',
        'has_token': bool(HF_TOKEN),
        'platform': 'Railway.app',
        'version': '1.0.0'
    })

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio file using Hugging Face API"""
    try:
        # Check if audio file is in request
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided. Send as multipart/form-data with field name "audio"'
            }), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(audio_file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not supported. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Get HF token
        token = HF_TOKEN or request.form.get('hf_token')
        if not token:
            return jsonify({
                'success': False,
                'error': 'Hugging Face token required. Set HF_TOKEN environment variable or pass hf_token in form data',
                'get_token': 'https://huggingface.co/settings/tokens'
            }), 400
        
        logger.info(f"🎵 Processing file: {audio_file.filename}")
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {token}"
        }
        
        # Read audio file data
        audio_data = audio_file.read()
        
        logger.info("🔄 Sending to Hugging Face API...")
        
        # Send request to Hugging Face
        response = requests.post(HF_API_URL, headers=headers, data=audio_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Transcription completed")
            
            # Handle different response formats
            text = ""
            if isinstance(result, dict):
                text = result.get('text', '')
            elif isinstance(result, list) and len(result) > 0:
                text = result[0].get('text', '') if isinstance(result[0], dict) else str(result[0])
            
            return jsonify({
                'success': True,
                'text': text,
                'model': 'openai/whisper-small',
                'backend': 'huggingface',
                'platform': 'Railway.app',
                'cost': 'FREE'
            })
        else:
            logger.error(f"❌ HF API error: {response.status_code} - {response.text}")
            error_msg = response.text
            if response.status_code == 401:
                error_msg = "Invalid Hugging Face token"
            elif response.status_code == 503:
                error_msg = "Model is loading, please try again in a few seconds"
            
            return jsonify({
                'success': False,
                'error': error_msg,
                'status_code': response.status_code
            }), response.status_code
            
    except Exception as e:
        logger.error(f"❌ Transcription error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Get model information"""
    return jsonify({
        'current_model': 'openai/whisper-small',
        'backend': 'huggingface',
        'type': 'proxy',
        'platform': 'Railway.app',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'features': [
            'Free Hugging Face API',
            'No local model required',
            'Multiple audio formats',
            'Fast processing'
        ]
    })

if __name__ == '__main__':
    # Get port from environment (Railway sets this)
    port = int(os.environ.get('PORT', 8000))
    
    logger.info("🚂 Starting Railway Whisper Proxy API")
    logger.info("=" * 50)
    logger.info("🎤 Service: Free Speech-to-Text API")
    logger.info("🔗 Backend: Hugging Face API")
    logger.info("🤖 Model: OpenAI Whisper Small")
    logger.info("☁️  Platform: Railway.app")
    logger.info("💰 Cost: FREE")
    logger.info(f"🌐 Port: {port}")
    logger.info("=" * 50)
    
    if not HF_TOKEN:
        logger.warning("⚠️  HF_TOKEN not set. Users must provide token in requests.")
    
    app.run(host='0.0.0.0', port=port, debug=False)
