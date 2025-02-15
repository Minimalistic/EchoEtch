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
import json
import re
from datetime import datetime

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
        self.failed_files = {}  # Track failed files and their attempt counts
        self.max_retry_attempts = 3  # Maximum number of retry attempts for failed files
        self.last_health_check = time.time()
        self.last_directory_scan = time.time()
        self.health_check_interval = 3600  # Run health check every hour
        self.directory_scan_interval = 30  # Scan directory every 5 minutes
        self.max_processed_files = 1000  # Maximum number of processed files to track
        self.files_in_progress = {}  # Track files that are being monitored for stability
        self.stability_check_interval = 1  # Check file stability every second
        self.required_stable_time = 3  # File must be stable for 3 seconds
        self.max_wait_time = 60  # Maximum time to wait for file stability (1 minute)
        self.last_empty_notification = 0  # Track when we last notified about empty folder
        self.empty_notification_interval = 30  # How often to notify about empty folder (10 minutes)
        
        # Ensure error directory exists
        self.error_dir = Path(os.getenv('WATCH_FOLDER')) / 'errors'
        self.error_dir.mkdir(exist_ok=True)
        
        # Initial scan
        self.scan_directory()
        
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

        # Check if it's time to scan the directory
        if current_time - self.last_directory_scan >= self.directory_scan_interval:
            had_files = self.scan_directory()
            self.last_directory_scan = current_time
            
            # Only log empty status periodically to avoid spam
            if not had_files and not self.files_in_progress:
                if current_time - self.last_empty_notification >= self.empty_notification_interval:
                    next_scan = time.localtime(current_time + self.directory_scan_interval)
                    next_check = time.strftime('%I:%M %p', next_scan)
                    logging.info(f"Watch folder is empty, monitoring for new files (next check at {next_check})")
                    self.last_empty_notification = current_time

        # Check files in progress
        if self.files_in_progress:
            self.check_files_in_progress()

    def check_ollama_health(self):
        """Check if Ollama is responsive"""
        try:
            response = requests.get(self.processor.api_url.replace('/api/generate', '/api/version'))
            return response.status_code == 200
        except:
            return False

    def check_files_in_progress(self):
        """Check the stability of files being monitored"""
        current_time = time.time()
        files_to_remove = []

        for file_path, file_info in self.files_in_progress.items():
            try:
                if not Path(file_path).exists():
                    files_to_remove.append(file_path)
                    continue

                current_size = Path(file_path).stat().st_size
                last_check_time = file_info['last_check_time']
                last_size = file_info['last_size']
                first_seen_time = file_info['first_seen_time']
                last_stable_time = file_info.get('last_stable_time', current_time)

                # Update size information
                if current_size != last_size:
                    file_info['last_size'] = current_size
                    file_info['last_stable_time'] = current_time
                elif current_time - last_stable_time >= self.required_stable_time:
                    # File has been stable for required time
                    logging.info(f"File {file_path} is stable, processing...")
                    self._process_audio_file(Path(file_path))
                    files_to_remove.append(file_path)
                elif current_time - first_seen_time >= self.max_wait_time:
                    # File has been waiting too long
                    logging.warning(f"File {file_path} exceeded maximum wait time, processing anyway...")
                    self._process_audio_file(Path(file_path))
                    files_to_remove.append(file_path)

                file_info['last_check_time'] = current_time

            except Exception as e:
                logging.error(f"Error checking file {file_path}: {str(e)}")
                files_to_remove.append(file_path)

        # Remove processed or errored files
        for file_path in files_to_remove:
            self.files_in_progress.pop(file_path, None)

    def on_created(self, event):
        if event.is_directory:
            return
        
        try:
            file_path = Path(event.src_path)
            if file_path.suffix.lower() in ['.mp3', '.wav', '.m4a']:
                # Check if we've already processed this file
                if file_path.name in self.processed_files:
                    logging.info(f"File already processed, skipping: {file_path}")
                    return
                
                logging.info(f"New audio file detected: {file_path}")
                self.start_monitoring_file(file_path)
                
        except Exception as e:
            logging.error(f"Error in on_created handler: {str(e)}")

    def start_monitoring_file(self, file_path):
        """Start monitoring a file for stability"""
        try:
            current_time = time.time()
            str_path = str(file_path)
            
            # Check if file has previously failed
            if str_path in self.failed_files:
                attempts = self.failed_files[str_path]
                if attempts >= self.max_retry_attempts:
                    logging.warning(f"File {file_path} has exceeded maximum retry attempts. Moving to error directory.")
                    self.move_to_error_dir(file_path)
                    return
                logging.info(f"Retrying file {file_path} (attempt {attempts + 1}/{self.max_retry_attempts})")
            
            if str_path not in self.files_in_progress:
                size = file_path.stat().st_size
                self.files_in_progress[str_path] = {
                    'first_seen_time': current_time,
                    'last_check_time': current_time,
                    'last_size': size,
                    'last_stable_time': current_time
                }
                logging.info(f"Started monitoring file: {file_path}")
        except Exception as e:
            logging.error(f"Error starting to monitor file {file_path}: {str(e)}")

    def move_to_error_dir(self, file_path):
        """Move a failed file to the error directory with metadata"""
        try:
            # Create a unique name in case of conflicts
            error_path = self.error_dir / f"{file_path.name}"
            if error_path.exists():
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                error_path = self.error_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            
            # Move the file
            file_path.rename(error_path)
            
            # Create metadata file
            metadata_path = error_path.with_suffix(error_path.suffix + '.error')
            with open(metadata_path, 'w') as f:
                metadata = {
                    'original_path': str(file_path),
                    'first_error_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'attempts': self.failed_files.get(str(file_path), 0),
                    'last_error': self.failed_files.get(str(file_path) + '_last_error', 'Unknown error')
                }
                json.dump(metadata, f, indent=2)
            
            logging.info(f"Moved failed file to {error_path} with metadata")
            
            # Remove from failed files tracking
            self.failed_files.pop(str(file_path), None)
            self.failed_files.pop(str(file_path) + '_last_error', None)
            
        except Exception as e:
            logging.error(f"Failed to move file {file_path} to error directory: {str(e)}")

    def _extract_source_datetime(self, file_path: Path) -> tuple:
        """Extract the source date and time from the audio filename"""
        try:
            # Extract date and optionally time from filename
            datetime_match = re.match(r'(\d{4}-\d{2}-\d{2})(?:[-_](\d{2}[-_]\d{2}(?:AM|PM)?))?\b', file_path.stem, re.IGNORECASE)
            if datetime_match:
                date = datetime_match.group(1)
                time = datetime_match.group(2) if datetime_match.group(2) else None
                return date, time
            return None, None
        except Exception as e:
            logging.error(f"Error extracting datetime from filename: {str(e)}")
            return None, None

    def _process_audio_file(self, file_path):
        try:
            logging.info(f"Processing file: {file_path}")
            
            str_path = str(file_path)
            
            # Check if file exists and is accessible
            if not file_path.exists():
                logging.debug(f"File not found (may have been moved): {file_path}")
                return
            
            # Extract source date and time
            source_date, source_time = self._extract_source_datetime(file_path)
            
            logging.info("Starting transcription...")
            transcription_data = self.transcriber.transcribe(file_path)
            logging.info("Transcription completed successfully")
            
            # Log metadata information
            if transcription_data.get("language"):
                logging.info(f"Detected language: {transcription_data['language']}")
            
            # Log confidence information
            low_confidence_segments = [s for s in transcription_data.get("segments", []) 
                                    if s.get("confidence", 0) < -1.0]
            if low_confidence_segments:
                logging.warning(f"Found {len(low_confidence_segments)} low confidence segments")
            
            # Process with Ollama
            processed_content = self.processor.process_transcription(transcription_data, file_path.name)
            
            # Add source date/time to processed content
            if source_date:
                processed_content['source_date'] = source_date
                if source_time:
                    processed_content['source_time'] = source_time
            
            logging.info("Ollama processing completed")
            
            # Create note
            self.note_manager.create_note(processed_content, file_path)
            logging.info(f"Note created successfully for: {file_path}")
            
            # Add to processed files
            self.processed_files.add(file_path.name)
            
            # Remove from failed files if it was there
            if str_path in self.failed_files:
                self.failed_files.pop(str_path)
                self.failed_files.pop(str_path + '_last_error', None)
                
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error processing {file_path}: {error_msg}")
            logging.exception("Full error trace:")
            
            str_path = str(file_path)
            # Track the failure
            self.failed_files[str_path] = self.failed_files.get(str_path, 0) + 1
            self.failed_files[str_path + '_last_error'] = error_msg
            
            # If we've exceeded max retries, move to error directory
            if self.failed_files[str_path] >= self.max_retry_attempts:
                logging.warning(f"File {file_path} has exceeded maximum retry attempts. Moving to error directory.")
                self.move_to_error_dir(file_path)

    def scan_directory(self):
        """Scan the watch directory for any unprocessed audio files"""
        try:
            watch_path = os.getenv('WATCH_FOLDER')
            if not watch_path or not os.path.exists(watch_path):
                logging.error("Watch folder not found or not set")
                return False

            # Get all files except those in the errors directory
            all_files = [f for f in Path(watch_path).glob('*') 
                        if f.parent != self.error_dir and f.is_file() 
                        and f.suffix.lower() in ['.mp3', '.wav', '.m4a']]
            
            if all_files:
                logging.info(f"Found {len(all_files)} audio files in watch directory")
                for file_path in all_files:
                    str_path = str(file_path)
                    if file_path.name not in self.processed_files and str_path not in self.files_in_progress:
                        logging.info(f"Found new file: {file_path.name}")
                        self.start_monitoring_file(file_path)
                return True
            return False
            
        except Exception as e:
            logging.error(f"Error during directory scan: {str(e)}")
            logging.exception("Full error trace:")
            return False

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
    load_dotenv(override=True)
    
    # Debug: Print environment variables
    logging.info(f"WATCH_FOLDER = {os.getenv('WATCH_FOLDER')}")
    logging.info(f"OBSIDIAN_VAULT_PATH = {os.getenv('OBSIDIAN_VAULT_PATH')}")
    logging.info(f"NOTES_FOLDER = {os.getenv('NOTES_FOLDER')}")
    
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
        
        # Log initial startup message
        logging.info(f"Started watching folder: {watch_path}")
        logging.info(f"Checking for new files every {event_handler.directory_scan_interval} seconds")
        
        # Main loop with health monitoring
        last_check = time.time()
        check_interval = event_handler.directory_scan_interval  # Match the handler's interval
        
        while True:
            try:
                time.sleep(1)  # Sleep for 1 second, no need for rapid checks
                
                current_time = time.time()
                if current_time - last_check >= check_interval:
                    # Force a health check and directory scan
                    event_handler.check_health()
                    last_check = current_time
                
                if not observer.is_alive():
                    logging.error("Observer thread died, restarting...")
                    observer.stop()
                    observer.join()
                    observer = Observer()
                    observer.schedule(event_handler, watch_path, recursive=False)
                    observer.start()
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                logging.exception("Full error trace:")
                time.sleep(5)  # Wait before retrying
                
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
