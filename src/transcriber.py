import whisper
from pathlib import Path
import torch
import re

class WhisperTranscriber:
    def __init__(self, model_size="base"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size).to(device)
        self.device = device

    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribe an audio file using Whisper
        
        Args:
            audio_path (Path): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            # Use better settings for GPU processing
            options = {
                "fp16": False,  # Use full precision for better quality
                "beam_size": 5,  # Increase beam size for better accuracy
                "best_of": 5,   # Consider more candidates
                "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Multiple temperatures for better results
            }
            
            result = self.model.transcribe(str(audio_path), **options)
            text = result["text"].strip()
            
            # Clean up the text
            # Remove multiple newlines
            text = re.sub(r'\n\s*\n', '\n', text)
            # Fix common punctuation issues
            text = re.sub(r'\s+([.,!?])', r'\1', text)
            # Ensure proper spacing after punctuation
            text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)
            # Remove any remaining excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")
