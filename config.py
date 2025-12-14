"""
Configuration settings for Wake Word Detection and Speaker Recognition system.
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# CEO voice embedding file
CEO_EMBEDDING_FILE = EMBEDDINGS_DIR / "ceo_voice.pkl"

# Audio settings
SAMPLE_RATE = 16000
FRAME_LENGTH = 512

# Vosk settings (No API key required!)
VOSK_MODEL_PATH = BASE_DIR / "vosk-model"  # Path to Vosk model
WAKE_PHRASE = "hello sara"  # The phrase to detect (case-insensitive)

# Alternative wake phrases to also detect (optional)
ALTERNATIVE_WAKE_PHRASES = ["hey sara", "hello sarah"]

# SpeechBrain model
SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# Voice Authentication Settings (pyannote.audio)
VOICE_THRESHOLD = 0.92  # Similarity threshold for authentication - EXTREMELY STRICT
# Recommended values for pyannote.audio embeddings:
# 0.75 = EXTREMELY STRICT (maximum security, only near-perfect matches)
# 0.70 = VERY STRICT (high security, recommended)
# 0.65 = Strict (good security)
# 0.60 = Balanced (moderate security)
# 0.55 = Lenient (easier to authenticate)
# 0.50 = Very lenient (low security)

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
