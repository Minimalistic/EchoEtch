import os
import time
from pathlib import Path
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from src.transcriber import WhisperTranscriber
from src.processor import OllamaProcessor
from src.note_manager import NoteManager

load_dotenv()

class AudioFileHandler(FileSystemEventHandler):
    def __init__(self):
        logging.info("Initializing AudioFileHandler...")
        try:
            self.transcriber = WhisperTranscriber()
            logging.info("Whisper model loaded successfully")
            self.processor = OllamaProcessor()
            logging.info("Ollama processor initialized")
            self.note_manager = NoteManager()
            logging.info("Note manager initialized")
        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            raise

    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix.lower() in ['.mp3', '.wav', '.m4a']:
            logging.info(f"New audio file detected: {file_path}")
            self._process_audio_file(file_path)

    def _process_audio_file(self, file_path):
        try:
            # Wait a moment to ensure file is fully written
            time.sleep(1)
            
            # Check if file exists and is accessible
            if not file_path.exists():
                logging.error(f"File does not exist: {file_path}")
                return
                
            logging.info("Starting transcription...")
            transcription = self.transcriber.transcribe(file_path)
            logging.info("Transcription completed successfully")
            
            logging.info("Processing with Ollama...")
            processed_content = self.processor.process_transcription(transcription)
            logging.info("Ollama processing completed")
            
            logging.info("Creating note...")
            self.note_manager.create_note(processed_content, file_path)
            logging.info(f"Note created successfully for: {file_path}")
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            logging.exception("Full error trace:")

def main():
    audio_folder = os.getenv('AUDIO_INPUT_FOLDER')
    if not audio_folder:
        logging.error("AUDIO_INPUT_FOLDER not set in .env file")
        return

    audio_folder_path = Path(audio_folder)
    if not audio_folder_path.exists():
        logging.error(f"Audio folder does not exist: {audio_folder}")
        return

    logging.info(f"Starting file monitor for: {audio_folder}")
    event_handler = AudioFileHandler()
    observer = Observer()
    observer.schedule(event_handler, audio_folder, recursive=False)
    observer.start()

    logging.info("File monitor started successfully")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("\nStopping folder monitoring...")
    observer.join()

if __name__ == "__main__":
    main()
