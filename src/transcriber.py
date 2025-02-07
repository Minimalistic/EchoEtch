import whisper
from pathlib import Path

class WhisperTranscriber:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribe an audio file using Whisper
        
        Args:
            audio_path (Path): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            result = self.model.transcribe(str(audio_path))
            return result["text"].strip()
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")
