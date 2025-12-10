"""
Deep Learning Voice Authenticator using SpeechBrain pre-trained models.
More accurate than MFCC approach, optimized with CPU-only PyTorch.
"""
import torch
import torchaudio
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Tuple
from speechbrain.inference.speaker import EncoderClassifier
import config

logger = logging.getLogger(__name__)


class VoiceAuthenticator:
    """
    Speaker verification using SpeechBrain's ECAPA-TDNN model.
    Pre-trained on VoxCeleb dataset for accurate speaker recognition.
    """
    
    def __init__(self, embedding_dir: str = "embeddings"):
        """
        Initialize the Voice Authenticator with deep learning model.
        
        Args:
            embedding_dir: Directory to store voice embeddings
        """
        self.embedding_dir = Path(embedding_dir)
        self.embedding_dir.mkdir(exist_ok=True)
        
        self.ceo_embedding_path = self.embedding_dir / "ceo_voice.pkl"
        self.ceo_embedding = None
        
        # Load pre-trained speaker recognition model (CPU-only)
        logger.info("Loading SpeechBrain speaker recognition model (CPU)...")
        try:
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"}
            )
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Load CEO embedding if exists
        if self.ceo_embedding_path.exists():
            self._load_ceo_embedding()
    
    def _load_ceo_embedding(self):
        """Load CEO voice embedding from disk."""
        try:
            with open(self.ceo_embedding_path, 'rb') as f:
                self.ceo_embedding = pickle.load(f)
            logger.info(f"✓ CEO embedding loaded from {self.ceo_embedding_path}")
        except Exception as e:
            logger.error(f"Failed to load CEO embedding: {e}")
            self.ceo_embedding = None
    
    def _save_ceo_embedding(self):
        """Save CEO voice embedding to disk."""
        try:
            with open(self.ceo_embedding_path, 'wb') as f:
                pickle.dump(self.ceo_embedding, f)
            logger.info(f"✓ CEO embedding saved to {self.ceo_embedding_path}")
        except Exception as e:
            logger.error(f"Failed to save CEO embedding: {e}")
    
    def create_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """
        Create speaker embedding from audio using deep learning.
        
        Args:
            audio: Audio waveform as numpy array (float32)
            sample_rate: Sample rate of audio
            
        Returns:
            Speaker embedding tensor
        """
        try:
            # Convert to tensor
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio
            
            # Ensure 1D
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.squeeze()
            
            # Normalize
            audio_tensor = audio_tensor / (torch.max(torch.abs(audio_tensor)) + 1e-8)
            
            # Add batch dimension
            audio_tensor = audio_tensor.unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(audio_tensor)
                embedding = embedding.squeeze()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise
    
    def enroll_speaker(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Enroll CEO voice by creating and saving deep learning embedding.
        
        Args:
            audio: Audio waveform as numpy array (float32)
            sample_rate: Sample rate of audio
            
        Returns:
            True if enrollment successful
        """
        try:
            # Check audio length (at least 2 seconds for better accuracy)
            min_length = sample_rate * 2.0
            if len(audio) < min_length:
                logger.error(f"Audio too short: {len(audio)/sample_rate:.2f}s (need at least 2s)")
                return False
            
            logger.info("Enrolling CEO voice with deep learning model...")
            
            # Create embedding
            embedding = self.create_embedding(audio, sample_rate)
            
            # Save as CEO embedding
            self.ceo_embedding = embedding
            self._save_ceo_embedding()
            
            logger.info("✓ CEO voice enrolled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Enrollment failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify_speaker(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[bool, float]:
        """
        Verify if audio matches enrolled CEO voice using deep learning.
        
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
            # Check audio length (at least 1 second)
            min_length = sample_rate * 1.0
            if len(audio) < min_length:
                logger.warning(f"Audio too short: {len(audio)/sample_rate:.2f}s (need at least 1s)")
                return False, 0.0
            
            # Create embedding for test audio
            test_embedding = self.create_embedding(audio, sample_rate)
            
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                self.ceo_embedding.unsqueeze(0),
                test_embedding.unsqueeze(0)
            ).item()
            
            # Convert from [-1, 1] to [0, 1]
            similarity = (similarity + 1) / 2
            
            # Use threshold from config
            threshold = config.VOICE_THRESHOLD
            is_ceo = similarity >= threshold
            
            logger.info(f"Deep Learning Similarity: {similarity:.4f} (threshold: {threshold})")
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
