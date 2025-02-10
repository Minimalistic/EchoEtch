import os
import time
from pathlib import Path
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from logging.handlers import RotatingFileHandler
import requests
import subprocess
import platform

# Set up logging configuration
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure the rotating file handler
    log_file = log_dir / "talknote.log"
    max_bytes = 10 * 1024 * 1024  # 10MB per file
    backup_count = 5  # Keep 5 backup files
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to the handlers
    log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get the root logger and set its level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers and add our configured handlers
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info("Logging system initialized with rotation enabled")

from src.transcriber import WhisperTranscriber
from src.processor import OllamaProcessor
from src.note_manager import NoteManager

class AudioFileHandler(FileSystemEventHandler):
    def __init__(self):
        setup_logging()
        logging.info("Initializing AudioFileHandler...")
        try:
            self.transcriber = WhisperTranscriber()
            logging.info("Whisper model loaded successfully")
            self.processor = OllamaProcessor()
            logging.info("Ollama processor initialized")
            self.note_manager = NoteManager()
            logging.info("Note manager initialized")
            self.processed_files = set()  # Track processed files
        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            raise

    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix.lower() in ['.mp3', '.wav', '.m4a']:
            # Check if we've already processed this file
            if file_path.name in self.processed_files:
                logging.info(f"File already processed, skipping: {file_path}")
                return
                
            logging.info(f"New audio file detected: {file_path}")
            self._process_audio_file(file_path)
            
            # Add to processed files set
            self.processed_files.add(file_path.name)
            
            # Cleanup old entries (optional, prevents unlimited growth)
            if len(self.processed_files) > 1000:
                self.processed_files.clear()

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

def ensure_ollama_running():
    """Check if Ollama is running and start it if not."""
    try:
        # Try to connect to Ollama API
        response = requests.get('http://localhost:11434/api/version')
        if response.status_code == 200:
            logging.info("Ollama is already running")
            return True
    except requests.exceptions.ConnectionError:
        logging.info("Ollama is not running. Attempting to start...")
        try:
            if platform.system() == 'Windows':
                # Start Ollama in a new process window
                subprocess.Popen('ollama serve', 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(['ollama', 'serve'])
            
            # Wait for Ollama to start (up to 30 seconds)
            max_attempts = 30
            for i in range(max_attempts):
                try:
                    response = requests.get('http://localhost:11434/api/version')
                    if response.status_code == 200:
                        logging.info("Ollama started successfully")
                        return True
                except requests.exceptions.ConnectionError:
                    if i < max_attempts - 1:
                        time.sleep(1)
                        continue
                    logging.error("Failed to start Ollama after 30 seconds")
                    return False
        except FileNotFoundError:
            logging.error("Ollama executable not found. Please ensure Ollama is installed")
            return False
        except Exception as e:
            logging.error(f"Error starting Ollama: {str(e)}")
            return False

def main():
    # Initialize logging first
    setup_logging()
    
    # Ensure Ollama is running before proceeding
    if not ensure_ollama_running():
        logging.error("Could not start Ollama. Exiting...")
        return

    # Load environment variables
    load_dotenv()
    
    audio_folder = os.getenv('AUDIO_INPUT_FOLDER')
    if not audio_folder:
        logging.error("AUDIO_INPUT_FOLDER not set in .env file")
        return

    # Normalize path and resolve any symlinks that might be present in iCloud Drive
    audio_folder_path = Path(audio_folder).resolve()
    logging.info(f"Resolved audio folder path: {audio_folder_path}")

    if not audio_folder_path.exists():
        logging.error(f"Audio folder does not exist: {audio_folder_path}")
        return

    if not audio_folder_path.is_dir():
        logging.error(f"Path exists but is not a directory: {audio_folder_path}")
        return

    # List current contents of the directory
    try:
        files = list(audio_folder_path.glob('*'))
        logging.info(f"Current files in directory: {[f.name for f in files]}")
    except Exception as e:
        logging.error(f"Error listing directory contents: {str(e)}")
        return

    logging.info(f"Starting file monitor for: {audio_folder_path}")
    event_handler = AudioFileHandler()
    observer = Observer()
    
    try:
        observer.schedule(event_handler, str(audio_folder_path), recursive=False)
        observer.start()
        logging.info("File monitor started successfully")
        
        while True:
            time.sleep(1)
    except Exception as e:
        logging.error(f"Error in file monitoring: {str(e)}")
        observer.stop()
    except KeyboardInterrupt:
        observer.stop()
        logging.info("\nStopping folder monitoring...")
    
    observer.join()

if __name__ == "__main__":
    main()
