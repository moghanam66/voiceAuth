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

# Voice Authentication Settings (MFCC-based lightweight system)
VOICE_THRESHOLD = 0.70  # Similarity threshold for authentication
# Recommended values:
# 0.65 = Lenient (easier to authenticate)
# 0.70 = Balanced (recommended)
# 0.75 = Strict (better security)
# 0.80 = Very strict (might reject legitimate users)

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
