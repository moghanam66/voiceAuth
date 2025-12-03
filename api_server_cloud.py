"""
Cloud-Ready REST API Server for Voice Authentication
Accepts audio file uploads for wake word detection and voice authentication.
No microphone/sounddevice required - works in Azure App Service.
"""
import os
import sys
import logging
import tempfile
import json
import wave
import numpy as np
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from vosk import Model, KaldiRecognizer

import config
from voice_authenticator import VoiceAuthenticator


# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Global components
vosk_model = None
voice_authenticator = None
wake_phrases = [config.WAKE_PHRASE.lower()] + [p.lower() for p in config.ALTERNATIVE_WAKE_PHRASES]


def initialize_models():
    """Initialize Vosk and Voice Authenticator models."""
    global vosk_model, voice_authenticator
    
    try:
        # Load Vosk model
        if not config.VOSK_MODEL_PATH.exists():
            logger.error(f"Vosk model not found at: {config.VOSK_MODEL_PATH}")
            logger.info("Downloading Vosk model... (this may take a few minutes)")
            # Auto-download if not present
            try:
                import download_vosk_model
                download_vosk_model.download_model()
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                return False
        
        logger.info(f"Loading Vosk model from: {config.VOSK_MODEL_PATH}")
        vosk_model = Model(str(config.VOSK_MODEL_PATH))
        logger.info("✓ Vosk model loaded")
        
        # Load voice authenticator
        logger.info("Loading voice authenticator...")
        voice_authenticator = VoiceAuthenticator()
        
        if not voice_authenticator.is_enrolled():
            logger.warning("WARNING: CEO voice not enrolled!")
            logger.warning("Voice authentication will not work until CEO voice is enrolled.")
        else:
            logger.info("✓ Voice authenticator loaded with CEO embedding")
        
        return True
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_for_wake_phrase(text: str) -> bool:
    """Check if text contains wake phrase."""
    text_lower = text.lower()
    
    # Fix common misrecognitions
    text_lower = text_lower.replace('lasorda', 'sara')
    text_lower = text_lower.replace('la sorda', 'sara')
    text_lower = text_lower.replace('sarah', 'sara')
    
    # Check for wake phrases
    for phrase in wake_phrases:
        if phrase in text_lower:
            return True
    
    # Fuzzy matching
    if 'hello' in text_lower and 'sara' in text_lower:
        return True
    if 'hey' in text_lower and 'sara' in text_lower:
        return True
    
    return False


def process_audio_file(file_path: str, sample_rate: int = 16000):
    """
    Process audio file for wake word detection and voice authentication.
    
    Args:
        file_path: Path to audio file (WAV format)
        sample_rate: Expected sample rate
        
    Returns:
        dict with detection results
    """
    try:
        # Read WAV file
        with wave.open(file_path, 'rb') as wf:
            if wf.getnchannels() != 1:
                return {'error': 'Audio must be mono (1 channel)'}
            if wf.getsampwidth() != 2:
                return {'error': 'Audio must be 16-bit PCM'}
            if wf.getframerate() != sample_rate:
                return {'error': f'Audio must be {sample_rate}Hz sample rate'}
            
            # Read all frames
            frames = wf.readframes(wf.getnframes())
        
        # Create recognizer
        recognizer = KaldiRecognizer(vosk_model, sample_rate)
        recognizer.SetWords(True)
        
        # Process audio
        recognizer.AcceptWaveform(frames)
        result = json.loads(recognizer.FinalResult())
        
        transcribed_text = result.get('text', '')
        wake_word_detected = check_for_wake_phrase(transcribed_text)
        
        # Voice authentication
        voice_authenticated = False
        similarity_score = 0.0
        
        if voice_authenticator and voice_authenticator.is_enrolled():
            # Convert bytes to numpy array (int16)
            audio_array = np.frombuffer(frames, dtype=np.int16)
            # Convert to float32 for voice authentication
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Verify speaker
            is_ceo, similarity = voice_authenticator.verify_speaker(
                audio_float,
                sample_rate=sample_rate
            )
            voice_authenticated = is_ceo
            similarity_score = similarity
        
        return {
            'transcribed_text': transcribed_text,
            'wake_word_detected': wake_word_detected,
            'voice_authenticated': voice_authenticated,
            'similarity_score': round(similarity_score, 4),
            'audio_duration_seconds': len(frames) / (sample_rate * 2)  # 2 bytes per sample
        }
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


@app.route('/')
def index():
    """Serve the frontend HTML."""
    try:
        return send_from_directory('static', 'index.html')
    except:
        # Fallback if static folder doesn't exist
        return jsonify({
            'service': 'Voice Authentication API',
            'version': '1.0.0',
            'message': 'Frontend not available. Use API endpoints directly.',
            'endpoints': {
                '/health': 'GET - Health check',
                '/process_audio': 'POST - Process audio file',
                '/enroll': 'POST - Enroll CEO voice'
            }
        })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    models_loaded = vosk_model is not None and voice_authenticator is not None
    ceo_enrolled = voice_authenticator.is_enrolled() if voice_authenticator else False
    
    return jsonify({
        'status': 'healthy' if models_loaded else 'initializing',
        'models_loaded': models_loaded,
        'ceo_enrolled': ceo_enrolled
    })


@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Process uploaded audio file for wake word detection and voice authentication.
    
    Expects:
        - Multipart form data with 'audio' file field
        - WAV format: 16kHz, mono, 16-bit PCM
        
    Returns:
        JSON with detection results
    """
    if vosk_model is None or voice_authenticator is None:
        return jsonify({'error': 'Models not initialized'}), 503
    
    # Check if file uploaded
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Save to temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        # Process audio
        result = process_audio_file(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error handling upload: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/enroll', methods=['POST'])
def enroll_ceo():
    """
    Enroll CEO voice from uploaded audio file.
    
    Expects:
        - Multipart form data with 'audio' file field
        - WAV format: 16kHz, mono, 16-bit PCM
        - At least 3-5 seconds of clear speech
        
    Returns:
        JSON with enrollment status
    """
    if voice_authenticator is None:
        return jsonify({'error': 'Voice authenticator not initialized'}), 503
    
    # Check if file uploaded
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Save to temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        # Read audio
        with wave.open(temp_path, 'rb') as wf:
            if wf.getnchannels() != 1:
                os.unlink(temp_path)
                return jsonify({'error': 'Audio must be mono (1 channel)'}), 400
            if wf.getsampwidth() != 2:
                os.unlink(temp_path)
                return jsonify({'error': 'Audio must be 16-bit PCM'}), 400
            if wf.getframerate() != 16000:
                os.unlink(temp_path)
                return jsonify({'error': 'Audio must be 16kHz sample rate'}), 400
            
            frames = wf.readframes(wf.getnframes())
        
        # Convert to numpy array
        audio_array = np.frombuffer(frames, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Enroll
        success = voice_authenticator.enroll_speaker(audio_float, sample_rate=16000)
        
        # Clean up
        os.unlink(temp_path)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'CEO voice enrolled successfully'
            })
        else:
            return jsonify({'error': 'Enrollment failed'}), 500
        
    except Exception as e:
        logger.error(f"Error during enrollment: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/info', methods=['GET'])
def api_info():
    """API information endpoint."""
    return jsonify({
        'service': 'Voice Authentication API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'GET - Frontend interface',
            '/health': 'GET - Health check',
            '/process_audio': 'POST - Process audio file (wake word + voice auth)',
            '/enroll': 'POST - Enroll CEO voice from audio file',
            '/api/info': 'GET - API information'
        },
        'audio_format': {
            'format': 'WAV',
            'sample_rate': '16000 Hz',
            'channels': 'mono (1)',
            'bit_depth': '16-bit PCM'
        }
    })


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("CLOUD VOICE AUTHENTICATION API SERVER")
    logger.info("Optimized for Azure App Service Deployment")
    logger.info("=" * 80)
    logger.info("")
    
    # Initialize models
    logger.info("Initializing models...")
    if not initialize_models():
        logger.error("Failed to initialize models. Exiting.")
        sys.exit(1)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("API ENDPOINTS:")
    logger.info("=" * 80)
    logger.info("  GET  /              - Frontend web interface")
    logger.info("  GET  /health        - Health check")
    logger.info("  POST /process_audio - Process audio file")
    logger.info("  POST /enroll        - Enroll CEO voice")
    logger.info("  GET  /api/info      - API information")
    logger.info("")
    logger.info("Starting Flask server on http://0.0.0.0:5000")
    logger.info("=" * 80)
    logger.info("")
    
    # Get port from environment (Azure sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Start Flask server
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
