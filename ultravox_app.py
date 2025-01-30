import gradio as gr
import torch
import transformers
import librosa
import numpy as np
import tempfile
import os

class UltravoxInterface:
    def __init__(self):
        """Initialize the Ultravox model and settings"""
        print("Loading Ultravox model... This may take a few minutes...")
        self.pipe = transformers.pipeline(
            model='fixie-ai/ultravox-v0_4',
            trust_remote_code=True
        )
        print("Model loaded successfully!")
        
        # Default system prompt
        self.default_prompt = "You are a friendly and helpful character. You love to answer questions for people."
        
    def process_audio(self, audio_path, custom_prompt=None):
        """
        Process audio input and return model response
        
        Args:
            audio_path: Path to the audio file
            custom_prompt: Optional custom system prompt
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
                max_new_tokens=30
            )
            
            # The output format changed in v0_4 - handle it directly
            if isinstance(result, str):
                return result
            elif isinstance(result, list):
                return result[0] if result else "No response generated"
            elif isinstance(result, dict):
                return result.get('generated_text', "No response generated")
            else:
                return str(result)
            
        except Exception as e:
            return f"Error processing audio: {str(e)}\nType of result: {type(result)}"

    def create_interface(self):
        """Create and configure the Gradio interface"""
        
        with gr.Blocks(title="Ultravox Voice Interface") as interface:
            gr.Markdown("# 🎙️ Ultravox Voice Assistant")
            gr.Markdown("Speak into the microphone and get AI-generated responses!")
            
            with gr.Row():
                with gr.Column():
                    # Updated Audio input component
                    audio_input = gr.Audio(
                        label="Speak here",
                        sources=["microphone"],  # Changed from source to sources
                        type="filepath"
                    )
                    
                    # Optional system prompt
                    system_prompt = gr.Textbox(
                        label="System Prompt (Optional)",
                        placeholder="Enter custom system prompt or leave empty for default",
                        value=self.default_prompt
                    )
                    
                    # Submit button
                    submit_btn = gr.Button("Process Audio", variant="primary")
                
                with gr.Column():
                    # Output text area
                    output_text = gr.Textbox(
                        label="AI Response",
                        lines=5,
                        placeholder="AI response will appear here..."
                    )
            
            # Handle submission
            submit_btn.click(
                fn=self.process_audio,
                inputs=[audio_input, system_prompt],
                outputs=output_text
            )
            
            # Example usage instructions
            gr.Markdown("""
            ## How to use:
            1. Click the microphone icon and allow browser access
            2. Speak your question or prompt
            3. Click 'Stop' when finished
            4. Click 'Process Audio' to get the AI response
            
            ## Requirements:
            - GPU with 24GB+ VRAM recommended
            - Working microphone
            - Stable internet connection
            
            ## Note:
            First-time loading may take a few minutes as the model is downloaded.
            """)
            
        return interface

def main():
    # Create instance of UltravoxInterface
    app = UltravoxInterface()
    
    # Launch the interface
    interface = app.create_interface()
    interface.launch(
        share=True,  # Enable sharing via Gradio
        server_name="0.0.0.0",  # Make available on all network interfaces
        server_port=7860  # Default Gradio port
    )

if __name__ == "__main__":
    main()