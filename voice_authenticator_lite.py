"""
Enhanced Lightweight Voice Authenticator using multiple acoustic features.
Uses MFCC, spectral features, and statistical modeling for better accuracy.
No PyTorch/SpeechBrain required - uses librosa and scikit-learn.
"""
import os
import pickle
import logging
from pathlib import Path
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import config

logger = logging.getLogger(__name__)


class VoiceAuthenticator:
    """
    Enhanced speaker verification using multiple acoustic features.
    Combines MFCCs, spectral features, and prosodic information.
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
        self.scaler = StandardScaler()
        
        # Enhanced feature extraction parameters
        self.n_mfcc = 20  # MFCCs
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128  # Mel bands for spectral features
        
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
        Extract comprehensive acoustic features from audio.
        
        Args:
            audio: Audio waveform as numpy array (float32)
            sample_rate: Sample rate of audio
            
        Returns:
            Feature vector with MFCCs, spectral, and prosodic features
        """
        try:
            # 1. MFCC features (timbre characteristics)
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Delta and delta-delta for temporal dynamics
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # 2. Spectral features (voice quality)
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
            )
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # 3. Mel-frequency features
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio, sr=sample_rate, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length
            )
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # 4. Zero crossing rate (voicing characteristics)
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)
            
            # 5. Chroma features (pitch content)
            chroma = librosa.feature.chroma_stft(
                y=audio, sr=sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # Combine all features
            all_features = np.vstack([
                mfccs,
                mfcc_delta,
                mfcc_delta2,
                spectral_centroids,
                spectral_rolloff,
                spectral_bandwidth,
                zcr,
                chroma,
                mel_spectrogram_db[:20]  # First 20 mel bands
            ])
            
            # Compute statistical moments across time
            mean = np.mean(all_features, axis=1)
            std = np.std(all_features, axis=1)
            median = np.median(all_features, axis=1)
            q1 = np.percentile(all_features, 25, axis=1)
            q3 = np.percentile(all_features, 75, axis=1)
            
            # Combine statistics
            feature_vector = np.concatenate([mean, std, median, q1, q3])
            
            # Robust normalization
            feature_vector = (feature_vector - np.mean(feature_vector)) / (np.std(feature_vector) + 1e-8)
            
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
    
    def verify_speaker(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[bool, float]:
        """
        Verify if audio matches enrolled CEO voice using enhanced similarity metrics.
        
        Args:
            audio: Audio waveform as numpy array (float32)
            sample_rate: Sample rate of audio
            
        Returns:
            Tuple of (is_ceo, similarity_score)
        """
        if self.ceo_embedding is None:
            logger.warning("CEO voice not enrolled")
            return False, 0.0
        
        try:
            # Check audio length (minimum 1 second)
            min_length = sample_rate * 1.0
            if len(audio) < min_length:
                logger.warning(f"Audio too short: {len(audio)/sample_rate:.2f}s (need at least 1s)")
                return False, 0.0
            
            # Extract features
            features = self.extract_features(audio, sample_rate)
            
            # Multiple similarity metrics for better accuracy
            
            # 1. Cosine similarity (angle between vectors)
            cos_sim = cosine_similarity(
                features.reshape(1, -1),
                self.ceo_embedding.reshape(1, -1)
            )[0][0]
            
            # 2. Euclidean distance (normalized inverse)
            euclidean_dist = np.linalg.norm(features - self.ceo_embedding)
            max_dist = np.sqrt(len(features))  # Maximum possible distance
            euclidean_sim = 1 - (euclidean_dist / max_dist)
            
            # 3. Pearson correlation
            pearson_corr = np.corrcoef(features, self.ceo_embedding)[0, 1]
            pearson_sim = (pearson_corr + 1) / 2  # Convert to 0-1 range
            
            # Weighted combination of similarities
            similarity = (
                0.5 * cos_sim +           # Cosine (most important)
                0.3 * euclidean_sim +     # Euclidean distance
                0.2 * pearson_sim         # Correlation
            )
            
            # Ensure similarity is in [0, 1] range
            similarity = np.clip(similarity, 0.0, 1.0)
            
            # Use threshold from config
            threshold = config.VOICE_THRESHOLD
            is_ceo = similarity >= threshold
            
            logger.info(f"Similarity breakdown:")
            logger.info(f"  - Cosine: {cos_sim:.4f}")
            logger.info(f"  - Euclidean: {euclidean_sim:.4f}")
            logger.info(f"  - Pearson: {pearson_sim:.4f}")
            logger.info(f"  - Combined: {similarity:.4f} (threshold: {threshold})")
            logger.info(f"Result: {'✓ CEO VERIFIED' if is_ceo else '✗ NOT AUTHORIZED'}")
            
            return is_ceo, float(similarity)
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0
    
    def is_enrolled(self) -> bool:
        """Check if CEO voice is enrolled."""
        return self.ceo_embedding is not None
