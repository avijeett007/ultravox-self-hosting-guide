# live_speech_assistant.py

import tkinter as tk
from tkinter import ttk, scrolledtext
import torch
import transformers
import librosa
import numpy as np
import tempfile
import os
import webrtcvad
import pyaudio
import wave
import threading
import queue
import time
from kokoro import KPipeline
import soundfile as sf
from typing import Dict, Optional, Tuple, List
import collections
from datetime import datetime
import gradio as gr
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import json
import base64
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import struct
import torchaudio
import subprocess
import io

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000):
        print("Initializing Silero VAD...")
        print(f"Sample rate: {sample_rate}")
        
        self.sample_rate = sample_rate
        self.silent_frames = 0
        self.silent_threshold = 30
        self.min_speech_duration = 16000
        self.voiced_frames: List[torch.Tensor] = []
        self.is_speaking = False
        self.chunk_size = 512  # Silero VAD expects 512 samples for 16kHz
        
        # Load Silero VAD
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.model.eval()
        print("Silero VAD loaded successfully")

    def process_audio(self, audio_data: np.ndarray) -> tuple[Optional[np.ndarray], bool]:
        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(audio_data).float()
        
        # Split audio into chunks of 512 samples
        num_chunks = len(audio_tensor) // self.chunk_size
        speech_probs = []
        
        for i in range(num_chunks):
            chunk = audio_tensor[i * self.chunk_size:(i + 1) * self.chunk_size]
            # Get speech probability for this chunk
            prob = self.model(chunk, self.sample_rate).item()
            speech_probs.append(prob)
        
        # If there's a remainder, pad it to 512 samples
        if len(audio_tensor) % self.chunk_size:
            last_chunk = audio_tensor[num_chunks * self.chunk_size:]
            padded_chunk = torch.nn.functional.pad(
                last_chunk, 
                (0, self.chunk_size - len(last_chunk))
            )
            prob = self.model(padded_chunk, self.sample_rate).item()
            speech_probs.append(prob)
        
        # Average probability across chunks
        avg_speech_prob = sum(speech_probs) / len(speech_probs)
        is_speech = avg_speech_prob > 0.5
        
        print(f"VAD: Speech prob {avg_speech_prob:.2f}, "
              f"Buffer size: {len(self.voiced_frames)}, "
              f"Silent frames: {self.silent_frames}")
        
        if is_speech:
            self.voiced_frames.append(audio_tensor)
            self.silent_frames = 0
            self.is_speaking = True
            print("Still speaking... Buffer growing")
            return None, True
        else:
            if self.is_speaking:
                self.silent_frames += 1
                print(f"Silence detected: {self.silent_frames}/{self.silent_threshold}")
                if self.silent_frames > self.silent_threshold:
                    if len(self.voiced_frames) * len(audio_data) > self.min_speech_duration:
                        print(f"Speech complete! Frames: {len(self.voiced_frames)}")
                        # Concatenate all voiced frames
                        complete_audio = torch.cat(self.voiced_frames, dim=0).numpy()
                        self.voiced_frames = []
                        self.is_speaking = False
                        self.silent_frames = 0
                        return complete_audio, False
                    else:
                        print("Speech too short, discarding")
                        self.voiced_frames = []
                        self.is_speaking = False
                        self.silent_frames = 0
                        return None, False
            return None, False

class VoiceAssistant:
    VOICES = {
        "Bella (US Female)": {"code": "af_bella", "lang_code": "a"},
        "Nicole (US Female)": {"code": "af_nicole", "lang_code": "a"},
        "Michael (US Male)": {"code": "am_michael", "lang_code": "a"},
        "Emma (UK Female)": {"code": "bf_emma", "lang_code": "b"},
        "George (UK Male)": {"code": "bm_george", "lang_code": "b"}
    }

    def __init__(self):
        print("Loading Ultravox model...")
        self.pipe = transformers.pipeline(
            model='fixie-ai/ultravox-v0_4',
            trust_remote_code=True
        )
        self.tts_pipelines = {}
        self.current_voice = list(self.VOICES.keys())[0]
        self.vad = webrtcvad.Vad(3)
        print("Initialization complete!")

    def get_tts_pipeline(self, voice_name):
        if voice_name not in self.tts_pipelines:
            voice_config = self.VOICES[voice_name]
            self.tts_pipelines[voice_name] = KPipeline(lang_code=voice_config["lang_code"])
        return self.tts_pipelines[voice_name]

    def is_speech(self, audio_data, sample_rate=16000):
        try:
            return self.vad.is_speech(audio_data.tobytes(), sample_rate)
        except:
            return False

# Create FastAPI app
app = FastAPI(
    title="Ultravox Voice Assistant",
    description="A voice-based AI assistant",
    version="1.0.0"
)

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static files directory for web interface
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)

# Create HTML interface
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Ultravox WebSocket Server is Ready</title>
</body>
</html>
"""

# Serve HTML interface
@app.get("/")
async def get_html():
    return HTMLResponse(content=html_content)

# Global assistant instance
assistant = None

@app.on_event("startup")
async def startup_event():
    global assistant
    print("Starting server and initializing models...")
    assistant = VoiceAssistant()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")
    
    try:
        await websocket.send_json({
            'status': 'ready',
            'message': 'Server ready',
            'timestamp': datetime.now().isoformat(),
            'voices': list(VoiceAssistant.VOICES.keys())
        })
        
        while True:
            message = await websocket.receive_json()
            message_type = message.get('type')
            print(f"Received message type: {message_type}")
            
            if message_type == 'audio_data':
                try:
                    print("Processing audio data...")
                    audio_data = message.get('audio')
                    voice = message.get('voice', assistant.current_voice)
                    
                    if not audio_data:
                        print("No audio data received")
                        continue
                    
                    # Decode base64 to WebM
                    audio_bytes = base64.b64decode(audio_data)
                    
                    # Save WebM temporarily
                    temp_webm = os.path.join(tempfile.gettempdir(), f'input_{os.urandom(4).hex()}.webm')
                    temp_wav = os.path.join(tempfile.gettempdir(), f'input_{os.urandom(4).hex()}.wav')
                    
                    try:
                        # Save WebM file
                        with open(temp_webm, 'wb') as f:
                            f.write(audio_bytes)
                        
                        # Convert WebM to WAV using ffmpeg
                        subprocess.run([
                            'ffmpeg', '-i', temp_webm,
                            '-acodec', 'pcm_s16le',
                            '-ar', '16000',
                            '-ac', '1',
                            temp_wav
                        ], check=True, capture_output=True)
                        
                        # Read the WAV file
                        audio, _ = sf.read(temp_wav, dtype='float32')
                        
                        print(f"Converted audio shape: {audio.shape}")
                        
                        # Process with Ultravox
                        print("Processing with Ultravox...")
                        result = assistant.pipe({
                            'audio': audio,
                            'turns': [{"role": "system", "content": "You are Knotie, an ultra-concise AI assistant. CORE RULES: 1) NEVER exceed 20 words per response, 2) Use quick acknowledgments ('mm-hmm', 'got it', 'I see'), 3) Answer ONLY what's asked, NO EXPLANATIONS OF YOUR THINKING 4) When speaking code or technical terms, spell out punctuation (e.g. say 'dot' instead of period, 'dash' instead of hyphen). KNOWLEDGE: You're an expert in: - AI development tutorials - Practical coding demos - Latest AI tools & frameworks. When referencing channel: Direct to kno2gether.com for full resources. STYLE: Friendly but extremely brief. CRITICAL: If unsure about specific technical details, say 'Please check kno2gether.com for the latest info on that.'"}],
                            'sampling_rate': 16000
                        }, max_new_tokens=200)

                        text_response = result[0] if isinstance(result, list) else str(result)
                        print(f"Generated response: {text_response}")

                        # Generate speech response
                        print("Generating speech response...")
                        tts_pipeline = assistant.get_tts_pipeline(voice)
                        audio_segments = []
                        for _, _, audio_data in tts_pipeline(text_response, voice=assistant.VOICES[voice]["code"], speed=1):
                            audio_segments.append(audio_data)

                        if audio_segments:
                            combined_audio = np.concatenate(audio_segments)
                            temp_path = os.path.join(tempfile.gettempdir(), f'response_{os.urandom(4).hex()}.wav')
                            sf.write(temp_path, combined_audio, 24000)
                            
                            with open(temp_path, 'rb') as f:
                                audio_base64 = base64.b64encode(f.read()).decode('utf-8')
                            
                            os.unlink(temp_path)
                            print("Sending response back to client")
                            
                            await websocket.send_json({
                                'status': 'success',
                                'text': text_response,
                                'audio': audio_base64,
                                'timestamp': datetime.now().isoformat()
                            })
                    
                    finally:
                        # Cleanup temporary files
                        if os.path.exists(temp_webm):
                            os.unlink(temp_webm)
                        if os.path.exists(temp_wav):
                            os.unlink(temp_wav)
                
                except Exception as e:
                    print(f"Error processing audio: {str(e)}")
                    print(f"Error type: {type(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    await websocket.send_json({
                        'status': 'error',
                        'message': str(e),
                        'timestamp': datetime.now().isoformat()
                    })

    except Exception as e:
        print(f"WebSocket error: {str(e)}")

def main():
    print("Starting Voice Assistant Server...")
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()