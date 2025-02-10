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
import signal
import sys
import gc

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
        self.initialize_components()
        self.processed_files = set()  # Track processed files
        self.last_health_check = time.time()
        self.health_check_interval = 3600  # Run health check every hour
        self.max_processed_files = 1000  # Maximum number of processed files to track

    def initialize_components(self):
        """Initialize or reinitialize components with error handling"""
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

    def check_health(self):
        """Perform periodic health checks and cleanup"""
        current_time = time.time()
        if current_time - self.last_health_check >= self.health_check_interval:
            logging.info("Performing periodic health check...")
            try:
                # Clean up processed files set to prevent memory growth
                if len(self.processed_files) > self.max_processed_files:
                    logging.info("Cleaning up processed files tracking set")
                    self.processed_files.clear()

                # Check Ollama connection
                if not self.check_ollama_health():
                    logging.warning("Ollama connection issue detected, reinitializing processor")
                    self.processor = OllamaProcessor()

                # Force garbage collection
                gc.collect()
                
                self.last_health_check = current_time
                logging.info("Health check completed successfully")
            except Exception as e:
                logging.error(f"Error during health check: {str(e)}")

    def check_ollama_health(self):
        """Check if Ollama is responsive"""
        try:
            response = requests.get(self.processor.api_url.replace('/api/generate', '/api/version'))
            return response.status_code == 200
        except:
            return False

    def on_created(self, event):
        if event.is_directory:
            return
        
        try:
            self.check_health()  # Run periodic health check
            
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
                
        except Exception as e:
            logging.error(f"Error in on_created handler: {str(e)}")

    def _process_audio_file(self, file_path):
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Wait a moment to ensure file is fully written
                time.sleep(1)
                
                # Check if file exists and is accessible
                if not file_path.exists():
                    if attempt == 0:  # Only log on first attempt
                        logging.debug(f"File not found (may have been moved): {file_path}")
                    return
                
                # Check file size stability
                initial_size = file_path.stat().st_size
                time.sleep(1)
                if initial_size != file_path.stat().st_size:
                    logging.info(f"File still being written, retrying... (attempt {attempt + 1})")
                    continue
                
                logging.info("Starting transcription...")
                transcription = self.transcriber.transcribe(file_path)
                logging.info("Transcription completed successfully")
                
                logging.info("Processing with Ollama...")
                processed_content = self.processor.process_transcription(transcription, file_path.name)
                logging.info("Ollama processing completed")
                
                logging.info("Creating note...")
                self.note_manager.create_note(processed_content, file_path)
                logging.info(f"Note created successfully for: {file_path}")
                return  # Exit after successful processing
                
            except FileNotFoundError:
                if attempt == 0:  # Only log on first attempt
                    logging.debug(f"File not found (may have been moved): {file_path}")
                return
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")
                logging.exception("Full error trace:")
                if attempt < max_retries - 1:  # Don't sleep on last attempt
                    time.sleep(retry_delay)  # Wait before retrying

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
    load_dotenv()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}. Initiating graceful shutdown...")
        observer.stop()
        observer.join()
        logging.info("Shutdown complete")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Ensure Ollama is running
        ensure_ollama_running()
        
        # Initialize the event handler
        event_handler = AudioFileHandler()
        
        # Set up the observer with error handling
        observer = Observer()
        watch_path = os.getenv('WATCH_FOLDER')
        if not watch_path:
            raise ValueError("WATCH_FOLDER environment variable not set")
        
        observer.schedule(event_handler, watch_path, recursive=False)
        observer.start()
        logging.info(f"Started watching folder: {watch_path}")
        
        # Main loop with health monitoring
        while True:
            try:
                time.sleep(1)
                if not observer.is_alive():
                    logging.error("Observer thread died, restarting...")
                    observer.stop()
                    observer.join()
                    observer = Observer()
                    observer.schedule(event_handler, watch_path, recursive=False)
                    observer.start()
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
                
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
