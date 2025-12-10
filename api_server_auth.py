"""
Simplified Voice Authentication API - Authentication Only (No Wake Word)
Accepts audio file uploads for speaker verification only.
Optimized for Azure App Service with minimal dependencies.
"""
import os
import sys
import logging
import tempfile
import wave
import numpy as np
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import config
from voice_authenticator_deep import VoiceAuthenticator


# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Global components
voice_authenticator = None


def initialize_models():
    """Initialize Voice Authenticator."""
    global voice_authenticator
    
    try:
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


def process_audio_file(file_path: str, sample_rate: int = 16000):
    """
    Process audio file for voice authentication only.
    
    Args:
        file_path: Path to audio file (WAV format)
        sample_rate: Expected sample rate
        
    Returns:
        dict with authentication results
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
            # Convert numpy types to Python native types for JSON serialization
            voice_authenticated = bool(is_ceo)
            similarity_score = float(similarity)
        
        return {
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
            'message': 'Authentication Only - No Wake Word Detection',
            'endpoints': {
                '/health': 'GET - Health check',
                '/authenticate': 'POST - Authenticate speaker',
                '/enroll': 'POST - Enroll CEO voice'
            }
        })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    models_loaded = voice_authenticator is not None
    ceo_enrolled = voice_authenticator.is_enrolled() if voice_authenticator else False
    
    return jsonify({
        'status': 'healthy' if models_loaded else 'initializing',
        'models_loaded': models_loaded,
        'ceo_enrolled': ceo_enrolled,
        'mode': 'authentication_only'
    })


@app.route('/authenticate', methods=['POST'])
def authenticate():
    """
    Authenticate speaker from uploaded audio file.
    
    Expects:
        - Multipart form data with 'audio' file field
        - WAV format: 16kHz, mono, 16-bit PCM
        
    Returns:
        JSON with authentication results
    """
    try:
        if voice_authenticator is None:
            return jsonify({'error': 'Voice authenticator not initialized'}), 503
        
        if not voice_authenticator.is_enrolled():
            return jsonify({'error': 'CEO voice not enrolled. Use /enroll endpoint first.'}), 400
    except Exception as e:
        logger.error(f"Error checking authenticator: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    
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
        import traceback
        traceback.print_exc()
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
        'mode': 'authentication_only',
        'endpoints': {
            '/': 'GET - Frontend interface',
            '/health': 'GET - Health check',
            '/authenticate': 'POST - Authenticate speaker',
            '/enroll': 'POST - Enroll CEO voice',
            '/api/info': 'GET - API information'
        },
        'audio_format': {
            'format': 'WAV',
            'sample_rate': '16000 Hz',
            'channels': 'mono (1)',
            'bit_depth': '16-bit PCM'
        }
    })


# Initialize on module load (for Gunicorn)
logger.info("=" * 80)
logger.info("VOICE AUTHENTICATION API SERVER")
logger.info("Authentication Only - No Wake Word Detection")
logger.info("=" * 80)
logger.info("Initializing voice authenticator...")
try:
    initialize_models()
    logger.info("✓ Initialization complete")
except Exception as e:
    logger.error(f"✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()


def main():
    """Main entry point for direct execution."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("API ENDPOINTS:")
    logger.info("=" * 80)
    logger.info("  GET  /              - Frontend web interface")
    logger.info("  GET  /health        - Health check")
    logger.info("  POST /authenticate  - Authenticate speaker")
    logger.info("  POST /enroll        - Enroll CEO voice")
    logger.info("  GET  /api/info      - API information")
    logger.info("")
    logger.info("Starting Flask development server...")
    logger.info("=" * 80)
    
    # Get port from environment (Azure sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Start Flask server
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
