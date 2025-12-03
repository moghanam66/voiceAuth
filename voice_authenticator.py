"""
Voice Authenticator module for speaker verification using SpeechBrain.
"""
import torch
import torchaudio
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Optional, Tuple
from speechbrain.pretrained import EncoderClassifier

import config


class VoiceAuthenticator:
    """
    Handles speaker verification using SpeechBrain's ECAPA-TDNN model.
    Compares incoming voice against pre-enrolled CEO voice embedding.
    """
    
    def __init__(self, embedding_file: Path = config.CEO_EMBEDDING_FILE):
        """
        Initialize the Voice Authenticator.
        
        Args:
            embedding_file: Path to the pickle file containing CEO voice embedding
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.embedding_file = embedding_file
        self.ceo_embedding = None
        self.similarity_threshold = config.SIMILARITY_THRESHOLD
        
        # Load SpeechBrain speaker recognition model
        self.logger.info(f"Loading speaker recognition model: {config.SPEAKER_MODEL}")
        try:
            # Use absolute path to avoid symlink issues on Windows
            import tempfile
            import os
            savedir = os.path.join(tempfile.gettempdir(), "spkrec-ecapa-voxceleb")
            
            self.model = EncoderClassifier.from_hparams(
                source=config.SPEAKER_MODEL,
                savedir=savedir,
                run_opts={"device": "cpu"}
            )
            self.logger.info("Speaker recognition model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load speaker recognition model: {e}")
            self.logger.error("If using Windows, try running PowerShell as Administrator")
            raise
        
        # Load CEO embedding if exists
        if self.embedding_file.exists():
            self.load_ceo_embedding()
        else:
            self.logger.warning(f"CEO embedding file not found at {self.embedding_file}")
            self.logger.warning("Please enroll CEO voice using enroll_ceo.py")
    
    def create_embedding(self, audio_data: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """
        Create speaker embedding from audio data.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Speaker embedding tensor
        """
        try:
            # Convert numpy array to torch tensor
            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.from_numpy(audio_data).float()
            else:
                audio_tensor = audio_data
            
            # Ensure audio is 1D
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.squeeze()
            
            # Normalize audio
            audio_tensor = audio_tensor / (torch.max(torch.abs(audio_tensor)) + 1e-8)
            
            # Add batch dimension
            audio_tensor = audio_tensor.unsqueeze(0)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(audio_tensor)
            
            return embedding.squeeze()
            
        except Exception as e:
            self.logger.error(f"Error creating embedding: {e}")
            raise
    
    def enroll_speaker(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Enroll CEO voice by creating and saving embedding.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            True if enrollment successful, False otherwise
        """
        try:
            self.logger.info("Enrolling CEO voice...")
            
            # Create embedding
            embedding = self.create_embedding(audio_data, sample_rate)
            
            # Save embedding
            self.ceo_embedding = embedding
            with open(self.embedding_file, 'wb') as f:
                pickle.dump(embedding.cpu().numpy(), f)
            
            self.logger.info(f"CEO voice enrolled and saved to {self.embedding_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error enrolling speaker: {e}")
            return False
    
    def load_ceo_embedding(self) -> bool:
        """
        Load CEO voice embedding from file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(self.embedding_file, 'rb') as f:
                embedding_array = pickle.load(f)
            
            self.ceo_embedding = torch.from_numpy(embedding_array)
            self.logger.info(f"CEO embedding loaded from {self.embedding_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading CEO embedding: {e}")
            return False
    
    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (higher = more similar)
        """
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            embedding1.unsqueeze(0), 
            embedding2.unsqueeze(0)
        )
        return similarity.item()
    
    def verify_speaker(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Tuple[bool, float]:
        """
        Verify if the speaker is the enrolled CEO.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Tuple of (is_ceo: bool, similarity_score: float)
        """
        if self.ceo_embedding is None:
            self.logger.error("CEO embedding not loaded. Please enroll CEO voice first.")
            return False, 0.0
        
        try:
            # Create embedding for incoming audio
            speaker_embedding = self.create_embedding(audio_data, sample_rate)
            
            # Compute similarity with CEO embedding
            similarity = self.compute_similarity(speaker_embedding, self.ceo_embedding)
            
            # Check if similarity meets threshold
            is_ceo = similarity >= self.similarity_threshold
            
            self.logger.info(f"Speaker verification: similarity={similarity:.4f}, threshold={self.similarity_threshold}")
            
            if is_ceo:
                self.logger.info("✓ CEO verified - AUTHORIZED")
            else:
                self.logger.warning("✗ Speaker mismatch - UNAUTHORIZED PERSON")
            
            return is_ceo, similarity
            
        except Exception as e:
            self.logger.error(f"Error verifying speaker: {e}")
            return False, 0.0
    
    def is_enrolled(self) -> bool:
        """
        Check if CEO voice is enrolled.
        
        Returns:
            True if CEO embedding exists, False otherwise
        """
        return self.ceo_embedding is not None
