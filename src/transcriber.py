import whisper
from pathlib import Path
import torch
import re

class WhisperTranscriber:
    def __init__(self, model_size="base"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size).to(device)
        self.device = device
        # Default initial prompt for personal note-taking context
        self.default_prompt = (
            "This is a personal note or journal entry. "
            "The speaker is sharing thoughts, ideas, or reflections "
            "in a natural, conversational style. "
            "Maintain proper sentence structure and punctuation."
        )

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
                "temperature": [0.0, 0.2, 0.4],  # Use lower temperatures for more consistent results
                "task": "transcribe",  # Explicitly set to transcribe
                "initial_prompt": self.default_prompt,  # Add the initial prompt
                "condition_on_previous_text": True,  # Help maintain context
                "compression_ratio_threshold": 2.4,  # Stricter threshold for repetition
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
