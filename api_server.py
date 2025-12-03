"""
REST API Server for Wake Word Detection and Voice Authentication
Provides real-time status via HTTP endpoints.
"""
import sys
import logging
import threading
from flask import Flask, jsonify
from flask_cors import CORS

import config
from main_realtime import ContinuousVoiceSystem


# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global voice system instance
voice_system = None
system_thread = None


@app.route('/status', methods=['GET'])
def get_status():
    """Get current wake word and authentication status."""
    if voice_system is None:
        return jsonify({
            'error': 'Voice system not initialized',
            'wake_word': False,
            'voice_authenticated': False,
            'listening': '',
            'similarity_score': 0.0
        }), 503
    
    status = voice_system._get_status()
    
    return jsonify({
        'wake_word': status['wake_word'],
        'voice_authenticated': status['authenticated'],
        'listening': status['text'],
        'similarity_score': round(status['score'], 3)
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'running',
        'system_active': voice_system is not None and voice_system.running
    })


@app.route('/wake_word', methods=['GET'])
def get_wake_word():
    """Get only wake word status."""
    if voice_system is None:
        return jsonify({'wake_word': False}), 503
    
    status = voice_system._get_status()
    return jsonify({'wake_word': status['wake_word']})


@app.route('/voice_auth', methods=['GET'])
def get_voice_auth():
    """Get only voice authentication status."""
    if voice_system is None:
        return jsonify({'voice_authenticated': False}), 503
    
    status = voice_system._get_status()
    return jsonify({
        'voice_authenticated': status['authenticated'],
        'similarity_score': round(status['score'], 3)
    })


def start_voice_system():
    """Start the voice system in a background thread."""
    global voice_system
    
    try:
        logger.info("Starting voice detection system...")
        voice_system = ContinuousVoiceSystem()
        voice_system.start()
    except Exception as e:
        logger.error(f"Error starting voice system: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("WAKE WORD DETECTION API SERVER")
    logger.info("Real-time Voice Detection & Authentication Endpoints")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Starting voice detection system in background...")
    logger.info("")
    
    # Start voice system in separate thread
    global system_thread
    system_thread = threading.Thread(target=start_voice_system, daemon=True)
    system_thread.start()
    
    # Wait for system to initialize
    import time
    time.sleep(3)
    
    logger.info("=" * 80)
    logger.info("API ENDPOINTS:")
    logger.info("=" * 80)
    logger.info("  GET /status        - Full status (wake_word, voice_authenticated, listening, score)")
    logger.info("  GET /wake_word     - Wake word detection status only")
    logger.info("  GET /voice_auth    - Voice authentication status only")
    logger.info("  GET /health        - Health check")
    logger.info("")
    logger.info("Example usage:")
    logger.info("  curl http://localhost:5000/status")
    logger.info("  curl http://localhost:5000/wake_word")
    logger.info("  curl http://localhost:5000/voice_auth")
    logger.info("")
    logger.info("Starting Flask API server on http://0.0.0.0:5000")
    logger.info("=" * 80)
    logger.info("")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()