import gradio as gr
import torch
import transformers
import librosa
import numpy as np
import tempfile
import os
from kokoro import KPipeline
import soundfile as sf
from typing import Dict, Optional, Tuple

class VoiceAssistant:
    # Available voices with their configurations
    VOICES = {
        "Bella (US Female)": {"code": "af_bella", "lang_code": "a"},
        "Nicole (US Female)": {"code": "af_nicole", "lang_code": "a"},
        "Michael (US Male)": {"code": "am_michael", "lang_code": "a"},
        "Emma (UK Female)": {"code": "bf_emma", "lang_code": "b"},
        "George (UK Male)": {"code": "bm_george", "lang_code": "b"}
    }

    def __init__(self):
        """Initialize both Ultravox and Kokoro TTS models"""
        print("Loading Ultravox model... This may take a few minutes...")
        self.pipe = transformers.pipeline(
            model='fixie-ai/ultravox-v0_4',  # Updated to v0_4
            trust_remote_code=True
        )
        print("Model loaded successfully!")

        # Initialize TTS pipelines dictionary
        self.tts_pipelines: Dict[str, KPipeline] = {}
        print("TTS models will be loaded on demand...")

        # Default system prompt
        self.default_prompt = "You are a friendly and helpful character. You love to answer questions for people."

    def get_tts_pipeline(self, voice_name: str) -> KPipeline:
        """Get or create TTS pipeline for specified voice"""
        if voice_name not in self.tts_pipelines:
            voice_config = self.VOICES[voice_name]
            self.tts_pipelines[voice_name] = KPipeline(lang_code=voice_config["lang_code"])
        return self.tts_pipelines[voice_name]

    def process_speech(self, audio_path: str, voice_name: str, custom_prompt: Optional[str] = None) -> Tuple[str, str]:
        """
        Process speech input and generate speech output
        
        Args:
            audio_path: Path to the audio file
            voice_name: Name of the voice to use for TTS
            custom_prompt: Optional custom system prompt
        
        Returns:
            tuple: (text_response, audio_path)
        """
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Prepare conversation turns
            turns = [
                {
                    "role": "system",
                    "content": custom_prompt if custom_prompt else self.default_prompt
                }
            ]
            
            # Get model response
            result = self.pipe(
                {
                    'audio': audio,
                    'turns': turns,
                    'sampling_rate': sr
                },
                max_new_tokens=200  # Increased for more detailed responses
            )
            
            # Handle different response formats from v0_4
            if isinstance(result, str):
                text_response = result
            elif isinstance(result, list):
                text_response = result[0] if result else "No response generated"
            elif isinstance(result, dict):
                text_response = result.get('generated_text', "No response generated")
            else:
                text_response = str(result)

            # Generate speech from text using Kokoro
            voice_config = self.VOICES[voice_name]
            pipeline = self.get_tts_pipeline(voice_name)
            
            # Create temporary file for audio output
            temp_path = os.path.join(tempfile.gettempdir(), f'response_{os.urandom(4).hex()}.wav')
            
            # Generate and save audio
            generator = pipeline(
                text_response,
                voice=voice_config["code"],
                speed=1,
            )
            
            # Process all audio segments
            audio_segments = []
            for _, _, audio_data in generator:
                audio_segments.append(audio_data)
            
            # Combine all segments and save
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                sf.write(temp_path, combined_audio, 24000)
                return text_response, temp_path
            
            return text_response, None

        except Exception as e:
            error_msg = f"Error processing speech: {str(e)}"
            print(error_msg)  # For debugging
            return error_msg, None

    def create_interface(self):
        """Create and configure the Gradio interface"""
        
        with gr.Blocks(title="Speech-to-Speech AI Assistant") as interface:
            gr.Markdown("# 🎙️ Speech-to-Speech AI Assistant")
            gr.Markdown("Speak your question and get an AI-generated voice response!")
            
            with gr.Row():
                with gr.Column():
                    # Audio input component
                    audio_input = gr.Audio(
                        label="Speak here",
                        sources=["microphone"],
                        type="filepath"
                    )
                    
                    # Voice selection
                    voice_dropdown = gr.Dropdown(
                        choices=list(self.VOICES.keys()),
                        value=list(self.VOICES.keys())[0],
                        label="Select Voice for Response"
                    )
                    
                    # Optional system prompt
                    system_prompt = gr.Textbox(
                        label="System Prompt (Optional)",
                        placeholder="Enter custom system prompt or leave empty for default",
                        value=self.default_prompt
                    )
                    
                    # Submit button
                    submit_btn = gr.Button("Process Speech", variant="primary")
                
                with gr.Column():
                    # Text output
                    text_output = gr.Textbox(
                        label="AI Response (Text)",
                        lines=5,
                        placeholder="AI response will appear here..."
                    )
                    
                    # Audio output
                    audio_output = gr.Audio(
                        label="AI Response (Voice)",
                        type="filepath"
                    )
            
            # Handle submission
            submit_btn.click(
                fn=self.process_speech,
                inputs=[audio_input, voice_dropdown, system_prompt],
                outputs=[text_output, audio_output]
            )
            
            # Example usage instructions
            gr.Markdown("""
            ## How to use:
            1. Click the microphone icon and allow browser access
            2. Speak your question or prompt
            3. Click 'Stop' when finished
            4. Select your preferred voice for the response
            5. Click 'Process Audio' to get both text and voice responses
            
            ## Available Voices:
            - US English: Bella (F), Nicole (F), Michael (M)
            - UK English: Emma (F), George (M)
            
            ## Requirements:
            - GPU with 24GB+ VRAM recommended
            - Working microphone
            - Stable internet connection
            
            ## Note:
            - First-time loading may take a few minutes
            - Each voice model will be downloaded when first used
            - The system supports continuous conversation
            """)
            
        return interface

def main():
    # Create instance of VoiceAssistant
    app = VoiceAssistant()
    
    # Launch the interface
    interface = app.create_interface()
    interface.launch(
        share=True,  # Enable sharing via Gradio
        server_name="0.0.0.0",  # Make available on all network interfaces
        server_port=7860  # Default Gradio port
    )

if __name__ == "__main__":
    main()