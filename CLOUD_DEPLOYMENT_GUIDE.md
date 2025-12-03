# 🎤 Voice Authentication System - Cloud Deployment

## What Was Fixed

### ❌ Previous Issues:
1. **PyAudio dependency** - Required system libraries (PortAudio) not available in Azure App Service
2. **Sounddevice dependency** - Requires audio input devices (microphone) which don't exist on cloud servers
3. **Real-time audio capture** - Not suitable for web API deployment

### ✅ Solution:
Created a **cloud-ready API** that:
- ✅ Accepts **file uploads** instead of live microphone input
- ✅ Removed PyAudio and sounddevice dependencies
- ✅ Uses only libraries compatible with Azure App Service
- ✅ Includes a **beautiful web frontend** for easy testing

---

## 🚀 Quick Start

### Local Testing

1. **Start the cloud API server:**
   ```bash
   python api_server_cloud.py
   ```

2. **Open your browser:**
   ```
   http://localhost:5000
   ```

3. **Use the web interface to:**
   - Upload CEO voice sample (Enroll CEO Voice tab)
   - Test voice authentication (Process Audio tab)

---

## 📦 Updated Package Requirements

The new `requirements.txt` includes only cloud-compatible packages:

```
Flask==3.0.0
flask-cors==4.0.0
gunicorn==21.2.0
numpy==1.26.2
scipy==1.11.4
requests==2.31.0
vosk==0.3.45
torch==2.1.0
torchaudio==2.1.0
speechbrain==0.5.16
```

**Removed:** ~~pyaudio~~, ~~tensorflow~~ (too large, replaced with lighter alternatives)

---

## 🌐 Frontend Features

The web interface (`static/index.html`) provides:

### 1. **Process Audio Tab**
- Upload WAV files for voice authentication
- See real-time results:
  - ✅ Wake word detected: YES/NO
  - ✅ Voice authenticated: AUTHORIZED/UNAUTHORIZED
  - ✅ Similarity score percentage
  - ✅ Transcribed text

### 2. **Enroll CEO Voice Tab**
- Upload CEO voice sample (3-10 seconds of clear speech)
- Enroll voice for future authentication

### 3. **Drag & Drop Support**
- Simply drag WAV files onto the upload areas

### 4. **Beautiful UI**
- Modern gradient design
- Responsive layout
- Real-time feedback
- Status badges

---

## 🎵 Audio Format Requirements

All audio files must be:
- **Format:** WAV (uncompressed)
- **Sample Rate:** 16000 Hz (16 kHz)
- **Channels:** Mono (1 channel)
- **Bit Depth:** 16-bit PCM
- **Duration:** 3-10 seconds recommended

### How to Convert Audio Files

**Using FFmpeg:**
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav
```

**Using Audacity:**
1. Open your audio file
2. Tracks → Resample → 16000 Hz
3. Tracks → Mix → Mix Stereo Down to Mono
4. File → Export → Export as WAV → 16-bit PCM

---

## 🔌 API Endpoints

### 1. **GET /** 
Frontend web interface

### 2. **GET /health**
Health check
```json
{
  "status": "healthy",
  "models_loaded": true,
  "ceo_enrolled": true
}
```

### 3. **POST /process_audio**
Process audio file for wake word + voice authentication

**Request:**
```bash
curl -X POST http://localhost:5000/process_audio \
  -F "audio=@test.wav"
```

**Response:**
```json
{
  "transcribed_text": "hello sara how are you",
  "wake_word_detected": true,
  "voice_authenticated": true,
  "similarity_score": 0.8523,
  "audio_duration_seconds": 4.5
}
```

### 4. **POST /enroll**
Enroll CEO voice

**Request:**
```bash
curl -X POST http://localhost:5000/enroll \
  -F "audio=@ceo_sample.wav"
```

**Response:**
```json
{
  "status": "success",
  "message": "CEO voice enrolled successfully"
}
```

---

## ☁️ Azure Deployment

### Files Updated for Cloud Deployment:

1. **`api_server_cloud.py`** - New cloud-ready API server
2. **`requirements.txt`** - Updated with cloud-compatible packages
3. **`runtime.txt`** - Specifies Python 3.11
4. **`startup.sh`** - Updated to use new API server
5. **`deploy.ps1`** - Updated to use Python 3.11
6. **`deploy.sh`** - Updated to use Python 3.11
7. **`static/index.html`** - New web frontend

### Deploy to Azure:

```powershell
# Run the deployment script
.\deploy.ps1
```

Or manually:
```powershell
az webapp up --resource-group wakeword-rg --name <YOUR-APP-NAME> --runtime "PYTHON:3.11"
```

### After Deployment:

1. Visit your Azure URL: `https://YOUR-APP-NAME.azurewebsites.net`
2. You'll see the web interface
3. First, enroll CEO voice using the "Enroll CEO Voice" tab
4. Then test voice authentication using the "Process Audio" tab

---

## 📝 Testing

### Create Test Audio Files

**Using Python:**
```python
import wave
import numpy as np

# Generate 3 seconds of test audio
sample_rate = 16000
duration = 3
samples = np.random.randint(-32768, 32767, sample_rate * duration, dtype=np.int16)

# Save as WAV
with wave.open('test.wav', 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(samples.tobytes())
```

**Record from microphone (for local testing only):**
```python
import sounddevice as sd
import wave

duration = 5  # seconds
sample_rate = 16000

print("Recording...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()
print("Done!")

with wave.open('recording.wav', 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(audio.tobytes())
```

---

## 🎯 What's Next?

Your system is now **cloud-ready** and will deploy successfully to Azure! 🎉

### Features:
✅ No PyAudio dependency
✅ File upload API instead of microphone
✅ Beautiful web interface
✅ Wake word detection
✅ Voice authentication
✅ Python 3.11 compatible
✅ Azure App Service optimized

### Access Your App:
- **Locally:** http://localhost:5000
- **Azure:** https://YOUR-APP-NAME.azurewebsites.net

---

## 🆘 Troubleshooting

### Issue: "Models not initialized"
**Solution:** Wait 1-2 minutes for Vosk model to download on first run

### Issue: "CEO voice not enrolled"
**Solution:** Use the "Enroll CEO Voice" tab to upload a CEO voice sample

### Issue: "Audio format error"
**Solution:** Ensure your WAV file is 16kHz, mono, 16-bit PCM

### Issue: Frontend not showing
**Solution:** Make sure the `static/` folder exists with `index.html`

---

## 📧 Support

For issues or questions, check the Azure deployment logs:
```bash
az webapp log tail --resource-group wakeword-rg --name YOUR-APP-NAME
```

---

**Deployment should now work perfectly! 🚀**
