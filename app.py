#!/usr/bin/env python3
"""
Railway.app Whisper API Server
Free speech-to-text using Hugging Face Transformers
Optimized for Railway's free tier resources
"""

import os
import logging
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model loading
transcriber = None
MODEL_LOADED = False

def load_model():
    """Load the Whisper model (lazy loading to save memory)"""
    global transcriber, MODEL_LOADED
    
    if MODEL_LOADED:
        return transcriber
    
    try:
        logger.info("Loading Whisper model...")
        import torch
        from transformers import pipeline
        
        # Optimize for Railway's limited memory
        torch.set_num_threads(1)  # Reduce CPU usage
        
        # Use tiny model for Railway's free tier (39MB)
        transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=-1,  # Use CPU
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        MODEL_LOADED = True
        logger.info("✅ Model loaded successfully!")
        return transcriber
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return None

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'm4a', 'flac', 'ogg', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation"""
    return jsonify({
        'service': 'Free Whisper Speech-to-Text API',
        'status': 'online',
        'model': 'openai/whisper-tiny',
        'platform': 'Railway.app',
        'cost': 'FREE',
        'endpoints': {
            'health': 'GET /',
            'transcribe': 'POST /transcribe (multipart/form-data with "audio" field)',
            'models': 'GET /models'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'example_curl': 'curl -X POST -F "audio=@your_file.mp3" https://your-app.railway.app/transcribe'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'platform': 'Railway.app',
        'version': '1.0.0'
    })

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio file to text"""
    try:
        # Load model if not already loaded
        model = load_model()
        if not model:
            return jsonify({
                'success': False,
                'error': 'Model failed to load'
            }), 500
        
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
        
        logger.info(f"🎵 Processing file: {audio_file.filename}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_file.save(tmp_file.name)
            
            try:
                # Transcribe audio
                logger.info("🔄 Starting transcription...")
                result = model(tmp_file.name)
                
                logger.info("✅ Transcription completed")
                
                return jsonify({
                    'success': True,
                    'text': result['text'],
                    'model': 'openai/whisper-tiny',
                    'platform': 'Railway.app',
                    'cost': 'FREE'
                })
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file.name)
                
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
        'current_model': 'openai/whisper-tiny',
        'model_size': '39MB',
        'platform': 'Railway.app',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'features': [
            'Free forever',
            'No API keys required',
            'Multiple audio formats',
            'Fast processing'
        ]
    })

if __name__ == '__main__':
    # Get port from environment (Railway sets this)
    port = int(os.environ.get('PORT', 8000))
    
    logger.info("🚂 Starting Railway Whisper API Server")
    logger.info("=" * 50)
    logger.info("🎤 Service: Free Speech-to-Text API")
    logger.info("🤖 Model: OpenAI Whisper Tiny")
    logger.info("☁️  Platform: Railway.app")
    logger.info("💰 Cost: FREE")
    logger.info(f"🌐 Port: {port}")
    logger.info("=" * 50)
    
    app.run(host='0.0.0.0', port=port, debug=False)
