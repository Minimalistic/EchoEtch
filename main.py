import os
import time
from pathlib import Path
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.transcriber import WhisperTranscriber
from src.processor import OllamaProcessor
from src.note_manager import NoteManager

load_dotenv()

class AudioFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.transcriber = WhisperTranscriber()
        self.processor = OllamaProcessor()
        self.note_manager = NoteManager()

    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix.lower() in ['.mp3', '.wav', '.m4a']:
            print(f"New audio file detected: {file_path}")
            self._process_audio_file(file_path)

    def _process_audio_file(self, file_path):
        try:
            # Transcribe audio
            transcription = self.transcriber.transcribe(file_path)
            
            # Process with Ollama
            processed_content = self.processor.process_transcription(transcription)
            
            # Create and save note
            self.note_manager.create_note(processed_content, file_path)
            
            print(f"Successfully processed: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

def main():
    audio_folder = os.getenv('AUDIO_INPUT_FOLDER')
    Path(audio_folder).mkdir(parents=True, exist_ok=True)

    event_handler = AudioFileHandler()
    observer = Observer()
    observer.schedule(event_handler, audio_folder, recursive=False)
    observer.start()

    print(f"Monitoring folder: {audio_folder}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopping folder monitoring...")
    observer.join()

if __name__ == "__main__":
    main()
