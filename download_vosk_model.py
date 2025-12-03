"""
Auto-download Vosk model for Azure deployment.
"""
import os
import sys
import urllib.request
import zipfile
import shutil

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip"
MODEL_ZIP = "vosk-model.zip"
MODEL_DIR = "vosk-model"
EXTRACTED_DIR = "vosk-model-en-us-0.22"

def download_and_extract():
    """Download and extract Vosk model."""
    
    # Check if already exists
    if os.path.exists(MODEL_DIR):
        print(f"✓ Model already exists at {MODEL_DIR}")
        return True
    
    try:
        print(f"Downloading model from {MODEL_URL}...")
        print("This is a 1.8 GB file, it will take several minutes...")
        
        # Download with progress
        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = downloaded * 100 / total_size
            if block_num % 50 == 0:  # Update every 50 blocks
                print(f"Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)")
        
        urllib.request.urlretrieve(MODEL_URL, MODEL_ZIP, progress)
        print("✓ Download complete!")
        
        # Extract
        print(f"Extracting model...")
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall()
        print("✓ Extraction complete!")
        
        # Rename if needed
        if os.path.exists(EXTRACTED_DIR):
            print(f"Renaming {EXTRACTED_DIR} to {MODEL_DIR}...")
            shutil.move(EXTRACTED_DIR, MODEL_DIR)
        
        # Cleanup
        if os.path.exists(MODEL_ZIP):
            print("Cleaning up zip file...")
            os.remove(MODEL_ZIP)
        
        print(f"✓ Model ready at {MODEL_DIR}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    success = download_and_extract()
    sys.exit(0 if success else 1)
