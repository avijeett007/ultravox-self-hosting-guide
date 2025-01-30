import asyncio
import base64
import json
import os
import tempfile
import time
from datetime import datetime
from typing import Dict, Optional
from urllib.parse import urljoin

import numpy as np
import soundfile as sf
import transformers
from fastapi import FastAPI, WebSocket, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from kokoro import KPipeline
from pydantic import BaseModel
import torch

class VoiceAssistant:
    VOICES = {
        "Bella (US Female)": {"code": "af_bella", "lang_code": "a"},
        "Nicole (US Female)": {"code": "af_nicole", "lang_code": "a"},
        "Michael (US Male)": {"code": "am_michael", "lang_code": "a"},
        "Emma (UK Female)": {"code": "bf_emma", "lang_code": "b"},
        "George (UK Male)": {"code": "bm_george", "lang_code": "b"}
    }

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        print("Loading Ultravox model...")
        self.pipe = transformers.pipeline(
            model='fixie-ai/ultravox-v0_4',
            trust_remote_code=True
        )
        self.tts_pipelines: Dict[str, KPipeline] = {}
        self.current_voice = list(self.VOICES.keys())[0]
        self.system_prompt = system_prompt
        self.state = "idle"
        
        # Initialize Silero VAD
        print("Loading Silero VAD...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.vad_model.eval()
        
        # Updated VAD parameters
        self.sample_rate = 16000
        self.vad_window_size = 1536  # Increased window size (96ms)
        self.speech_threshold = 0.5   # Speech probability threshold
        self.min_speech_duration_ms = 250  # Minimum speech duration in ms
        self.min_silence_duration_ms = 400  # Minimum silence duration in ms
        
        # New buffers
        self.audio_window = []
        self.speech_probs = []
        self.current_speech = []
        self.speech_timestamps = []
        
        # Buffers
        self.vad_buffer = []
        self.speech_buffer = []
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_counter = 0

    def get_tts_pipeline(self, voice_name: str) -> KPipeline:
        if voice_name not in self.tts_pipelines:
            voice_config = self.VOICES[voice_name]
            self.tts_pipelines[voice_name] = KPipeline(lang_code=voice_config["lang_code"])
        return self.tts_pipelines[voice_name]

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Improved VAD detection using sliding windows and averaging"""
        try:
            # Add chunk to audio window
            self.audio_window.extend(audio_chunk.tolist())
            
            # Process complete windows
            while len(self.audio_window) >= self.vad_window_size:
                # Get window of correct size
                window = np.array(self.audio_window[:self.vad_window_size])
                self.audio_window = self.audio_window[self.vad_window_size:]
                
                # Convert to tensor
                tensor = torch.FloatTensor(window).unsqueeze(0)
                
                # Get speech probability
                speech_prob = self.vad_model(tensor, self.sample_rate).item()
                self.speech_probs.append(speech_prob)
                
                # Keep only recent probabilities (last 1 second)
                window_size = int(self.sample_rate / self.vad_window_size)
                if len(self.speech_probs) > window_size:
                    self.speech_probs.pop(0)
            
            # Calculate moving average of speech probabilities
            if not self.speech_probs:
                return False
                
            avg_speech_prob = sum(self.speech_probs) / len(self.speech_probs)
            return avg_speech_prob > self.speech_threshold
            
        except Exception as e:
            print(f"VAD error: {e}")
            return False

    def add_audio(self, audio_chunk: np.ndarray) -> bool:
        """Improved audio processing with better speech detection"""
        has_speech = self.is_speech(audio_chunk)
        
        # Calculate durations
        chunk_duration_ms = len(audio_chunk) * 1000 / self.sample_rate
        
        if has_speech:
            if not self.current_speech:  # Start of speech
                print("Speech started...")
            self.current_speech.extend(audio_chunk.tolist())
            self.silence_counter = 0
        else:
            if self.current_speech:  # Potential end of speech
                self.silence_counter += chunk_duration_ms
                self.current_speech.extend(audio_chunk.tolist())
            
        # Check if we should process the speech
        should_process = (
            len(self.current_speech) > 0 and
            len(self.current_speech) * 1000 / self.sample_rate >= self.min_speech_duration_ms and
            self.silence_counter >= self.min_silence_duration_ms
        )
        
        if should_process:
            print(f"Speech detected - duration: {len(self.current_speech) * 1000 / self.sample_rate:.0f}ms")
            self.audio_buffer = self.current_speech.copy()
            self.current_speech = []
            self.silence_counter = 0
            self.speech_probs = []
            return True
            
        # Reset if silence is too long
        if self.silence_counter > self.min_silence_duration_ms * 2:
            self.current_speech = []
            self.silence_counter = 0
            self.speech_probs = []
            
        return False

    def get_audio(self) -> np.ndarray:
        """Get accumulated audio and clear buffer"""
        audio = np.array(self.audio_buffer)
        self.audio_buffer = []
        return audio

class CallConfig(BaseModel):
    systemPrompt: str
    temperature: float = 0.8
    voice: Optional[str] = None
    medium: dict = {
        "serverWebSocket": {
            "inputSampleRate": 16000,
            "outputSampleRate": 16000,
            "clientBufferSizeMs": 30000
        }
    }
    selectedTools: list = []
    firstSpeaker: str = "FIRST_SPEAKER_AGENT"
    initialOutputMedium: str = "MESSAGE_MEDIUM_SPEECH"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global assistant instance
assistant = None

def get_base_url(request: Request) -> str:
    """Get base URL from request"""
    host = request.headers.get("host", "localhost:7860")
    scheme = request.headers.get("x-forwarded-proto", "http")
    return f"{scheme}://{host}"

@app.post("/api/calls")
async def create_call(
    config: CallConfig,
    request: Request,
    x_api_key: Optional[str] = Header(None)
):
    # Optional: Validate API key if needed
    if os.getenv("REQUIRE_API_KEY"):
        if not x_api_key or x_api_key != os.getenv("ULTRAVOX_API_KEY"):
            return {"error": "Invalid API key"}, 401

    global assistant
    assistant = VoiceAssistant(system_prompt=config.systemPrompt)
    if config.voice:
        assistant.current_voice = config.voice
    
    # Generate a unique call ID
    call_id = f"call_{int(time.time())}"
    
    # Construct join URL using the request's base URL
    base_url = get_base_url(request)
    ws_scheme = "wss" if base_url.startswith("https") else "ws"
    join_url = f"{ws_scheme}://{request.headers['host']}/api/calls/{call_id}/join"
    
    return {
        "callId": call_id,
        "joinUrl": join_url,
        "status": "success"
    }

@app.websocket("/api/calls/{call_id}/join")
async def join_call(websocket: WebSocket, call_id: str):
    if not assistant:
        await websocket.close(code=4000, reason="No active call")
        return

    await websocket.accept()
    print(f"WebSocket connection established for call {call_id}")
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "state",
            "state": "speaking",
            "timestamp": int(time.time() * 1000)
        })

        # Send initial greeting
        initial_text = "Welcome to Dr. Donut! How can I help you today?"
        await websocket.send_json({
            "type": "transcript",
            "role": "agent",
            "text": initial_text,
            "final": True,
            "timestamp": int(time.time() * 1000)
        })

        # Generate and send initial TTS audio
        tts_pipeline = assistant.get_tts_pipeline(assistant.current_voice)
        audio_segments = []
        for _, _, audio_data in tts_pipeline(initial_text, 
                                           voice=assistant.VOICES[assistant.current_voice]["code"], 
                                           speed=1):
            audio_segments.append(audio_data)

        if audio_segments:
            combined_audio = np.concatenate(audio_segments)
            audio_int16 = (combined_audio * 32768).astype(np.int16)
            await websocket.send_bytes(audio_int16.tobytes())

        # Switch to listening state
        await websocket.send_json({
            "type": "state",
            "state": "listening",
            "timestamp": int(time.time() * 1000)
        })

        # Adjust VAD parameters
        assistant.min_speech_frames = 5
        assistant.silence_frames = 10

        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                
                if "bytes" in message:
                    audio_data = message["bytes"]
                    audio_chunk = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    print(f"Received chunk size: {len(audio_chunk)}")  # Debug log
                    
                    if assistant.add_audio(audio_chunk):
                        print("Speech detected, processing...")
                        audio = assistant.get_audio()
                        print(f"Processing audio length: {len(audio)}")
                        
                        if len(audio) < 1600:  # Skip if too short
                            print("Audio too short, skipping")
                            continue
                        
                        await websocket.send_json({
                            "type": "state",
                            "state": "thinking",
                            "timestamp": int(time.time() * 1000)
                        })
                        
                        result = assistant.pipe({
                            'audio': audio,
                            'turns': [{"role": "system", "content": assistant.system_prompt}],
                            'sampling_rate': 16000
                        }, max_new_tokens=200)

                        text_response = result[0] if isinstance(result, list) else str(result)
                        print(f"Generated response: {text_response}")
                        
                        await websocket.send_json({
                            "type": "transcript",
                            "role": "agent",
                            "text": text_response,
                            "final": True,
                            "timestamp": int(time.time() * 1000)
                        })
                        
                        await websocket.send_json({
                            "type": "state",
                            "state": "speaking",
                            "timestamp": int(time.time() * 1000)
                        })
                        
                        audio_segments = []
                        for _, _, audio_data in tts_pipeline(text_response, 
                                                           voice=assistant.VOICES[assistant.current_voice]["code"], 
                                                           speed=1):
                            audio_segments.append(audio_data)

                        if audio_segments:
                            combined_audio = np.concatenate(audio_segments)
                            audio_int16 = (combined_audio * 32768).astype(np.int16)
                            await websocket.send_bytes(audio_int16.tobytes())
                        
                        await websocket.send_json({
                            "type": "state",
                            "state": "listening",
                            "timestamp": int(time.time() * 1000)
                        })
                elif "type" in message and message["type"] == "close":
                    print(f"Client requested close for call {call_id}")
                    break

            except asyncio.TimeoutError:
                # Check if client is still connected
                try:
                    await websocket.send_json({"type": "ping"})
                except:
                    print(f"Client disconnected (timeout) for call {call_id}")
                    break
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                print(f"Current buffer sizes - VAD: {len(assistant.vad_buffer)}, Speech: {len(assistant.speech_buffer)}")  # Debug log
                if "disconnect" in str(e).lower() or "closed" in str(e).lower():
                    print(f"Client disconnected for call {call_id}")
                    break
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e),
                        "timestamp": int(time.time() * 1000)
                    })
                except:
                    break

    except Exception as e:
        print(f"WebSocket connection error for call {call_id}: {str(e)}")
    finally:
        print(f"Cleaning up call {call_id}")
        try:
            await websocket.close()
        except:
            pass
        # Clear assistant's state
        assistant.audio_buffer = []
        assistant.is_speaking = False
        assistant.speech_buffer = []

def main():
    print("Starting Speech Server...")
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=7860,
        ssl_keyfile=os.getenv("SSL_KEYFILE"),
        ssl_certfile=os.getenv("SSL_CERTFILE")
    )

if __name__ == "__main__":
    main() 