"""
Lightweight Voice Authenticator using MFCC features and cosine similarity.
No PyTorch/SpeechBrain required - uses librosa and scikit-learn instead.
Much smaller package footprint (~200MB vs ~2GB).
"""
import os
import pickle
import logging
from pathlib import Path
import numpy as np
import librosa
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class VoiceAuthenticator:
    """
    Lightweight speaker verification using MFCC features.
    Uses traditional signal processing instead of deep learning.
    """
    
    def __init__(self, embedding_dir: str = "embeddings"):
        """
        Initialize the voice authenticator.
        
        Args:
            embedding_dir: Directory to store voice embeddings
        """
        self.embedding_dir = Path(embedding_dir)
        self.embedding_dir.mkdir(exist_ok=True)
        
        self.ceo_embedding_path = self.embedding_dir / "ceo_voice.pkl"
        self.ceo_embedding = None
        
        # MFCC parameters
        self.n_mfcc = 40  # Number of MFCC coefficients
        self.n_fft = 2048
        self.hop_length = 512
        
        # Load CEO embedding if exists
        if self.ceo_embedding_path.exists():
            self._load_ceo_embedding()
    
    def _load_ceo_embedding(self):
        """Load CEO voice embedding from disk."""
        try:
            with open(self.ceo_embedding_path, 'rb') as f:
                self.ceo_embedding = pickle.load(f)
            logger.info(f"Loaded CEO embedding from {self.ceo_embedding_path}")
        except Exception as e:
            logger.error(f"Failed to load CEO embedding: {e}")
            self.ceo_embedding = None
    
    def _save_ceo_embedding(self):
        """Save CEO voice embedding to disk."""
        try:
            with open(self.ceo_embedding_path, 'wb') as f:
                pickle.dump(self.ceo_embedding, f)
            logger.info(f"Saved CEO embedding to {self.ceo_embedding_path}")
        except Exception as e:
            logger.error(f"Failed to save CEO embedding: {e}")
    
    def extract_features(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio waveform as numpy array (float32)
            sample_rate: Sample rate of audio
            
        Returns:
            Feature vector (normalized MFCC statistics)
        """
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Extract delta and delta-delta features
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Combine all features
            features = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
            
            # Compute statistics (mean and std) across time
            mean = np.mean(features, axis=1)
            std = np.std(features, axis=1)
            
            # Concatenate statistics
            feature_vector = np.concatenate([mean, std])
            
            # L2 normalization
            feature_vector = normalize(feature_vector.reshape(1, -1))[0]
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    
    def enroll_speaker(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Enroll CEO voice by extracting and saving features.
        
        Args:
            audio: Audio waveform as numpy array (float32)
            sample_rate: Sample rate of audio
            
        Returns:
            True if enrollment successful
        """
        try:
            # Check audio length (at least 1 second)
            min_length = sample_rate * 1.0
            if len(audio) < min_length:
                logger.error(f"Audio too short: {len(audio)/sample_rate:.2f}s (need at least 1s)")
                return False
            
            # Extract features
            logger.info("Extracting voice features for enrollment...")
            features = self.extract_features(audio, sample_rate)
            
            # Save as CEO embedding
            self.ceo_embedding = features
            self._save_ceo_embedding()
            
            logger.info("✓ CEO voice enrolled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Enrollment failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify_speaker(self, audio: np.ndarray, sample_rate: int = 16000, 
                      threshold: float = 0.75) -> tuple[bool, float]:
        """
        Verify if audio matches enrolled CEO voice.
        
        Args:
            audio: Audio waveform as numpy array (float32)
            sample_rate: Sample rate of audio
            threshold: Similarity threshold (0-1, higher = stricter)
            
        Returns:
            Tuple of (is_ceo, similarity_score)
        """
        if self.ceo_embedding is None:
            logger.warning("CEO voice not enrolled")
            return False, 0.0
        
        try:
            # Check audio length
            min_length = sample_rate * 0.5
            if len(audio) < min_length:
                logger.warning(f"Audio too short: {len(audio)/sample_rate:.2f}s")
                return False, 0.0
            
            # Extract features
            features = self.extract_features(audio, sample_rate)
            
            # Compute cosine similarity
            similarity = cosine_similarity(
                features.reshape(1, -1),
                self.ceo_embedding.reshape(1, -1)
            )[0][0]
            
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            similarity = (similarity + 1) / 2
            
            is_ceo = similarity >= threshold
            
            logger.info(f"Similarity: {similarity:.4f} (threshold: {threshold})")
            logger.info(f"Result: {'✓ CEO verified' if is_ceo else '✗ Not CEO'}")
            
            return is_ceo, float(similarity)
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0
    
    def is_enrolled(self) -> bool:
        """Check if CEO voice is enrolled."""
        return self.ceo_embedding is not None
