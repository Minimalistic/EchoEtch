import os
from datetime import datetime
from pathlib import Path
from typing import Dict
import re
import logging
import shutil

class NoteManager:
    def __init__(self):
        self.vault_path = Path(os.getenv('OBSIDIAN_VAULT_PATH'))
        self.notes_folder = self.vault_path / os.getenv('NOTES_FOLDER', 'daily_notes')
        self.notes_folder.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to remove invalid Windows characters
        """
        # Remove invalid Windows filename characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace multiple spaces with single space
        sanitized = re.sub(r'\s+', ' ', sanitized)
        # Trim spaces from ends
        return sanitized.strip()

    def _get_daily_folder(self) -> Path:
        """
        Get or create a daily folder for storing audio files
        Returns:
            Path: Path to the daily folder
        """
        today = datetime.now().strftime("%Y-%m-%d")
        daily_folder = self.vault_path / os.getenv('NOTES_FOLDER', 'daily_notes') / today / "audio"
        daily_folder.mkdir(parents=True, exist_ok=True)
        return daily_folder

    def _format_title(self, title: str) -> str:
        """
        Format a title by replacing hyphens with spaces and capitalizing words
        """
        # Replace hyphens with spaces
        title = title.replace('-', ' ')
        # Capitalize words, being careful with articles and conjunctions
        return ' '.join(word.capitalize() if word.lower() not in ['a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'at', 'by', 'in', 'of', 'on', 'to', 'up', 'as'] or i == 0 
                       else word.lower() 
                       for i, word in enumerate(title.split()))

    def create_note(self, processed_content: Dict, audio_file: Path):
        """
        Create a markdown note in the Obsidian vault
        
        Args:
            processed_content (Dict): Processed content from Ollama
            audio_file (Path): Path to the original audio file
        """
        # First, handle the audio file move
        daily_folder = self._get_daily_folder()
        
        # Create new audio filename with date, time and description
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%I-%M%p")
        title = processed_content.get('title', 'Untitled Note')
        sanitized_title = self._sanitize_filename(title[:30])
        new_audio_name = f"{date_time}_{sanitized_title}{audio_file.suffix}"
        new_audio_path = daily_folder / new_audio_name
        
        # Move audio file to daily folder if it's not already there
        if audio_file.parent != daily_folder:
            new_audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle case where file already exists
            if new_audio_path.exists():
                base = new_audio_path.stem
                suffix = new_audio_path.suffix
                counter = 1
                while new_audio_path.exists():
                    new_audio_path = daily_folder / f"{base}_{counter}{suffix}"
                    counter += 1
            
            logging.info(f"Moving audio file from {audio_file} to {new_audio_path}")
            
            try:
                # Copy file first
                shutil.copy2(str(audio_file), str(new_audio_path))
                
                # Verify copy was successful
                if new_audio_path.exists() and new_audio_path.stat().st_size == audio_file.stat().st_size:
                    # Only delete original after successful copy
                    audio_file.unlink()
                    logging.info("Audio file moved successfully")
                else:
                    raise Exception("File copy verification failed")
                
            except Exception as e:
                logging.error(f"Failed to move audio file: {str(e)}")
                raise Exception(f"Failed to move audio file: {str(e)}")
        
        # Create the note with matching naming convention
        note_filename = f"{date_time}_{sanitized_title}.md"
        note_path = daily_folder.parent / note_filename
        
        # Create relative link to audio file
        audio_rel_path = os.path.relpath(new_audio_path, self.vault_path)
        audio_rel_path = audio_rel_path.replace('\\', '/')  # Convert Windows path to forward slashes for markdown
        
        # Build note content
        note_content = []
        
        # Add title and tags at the top - use clean title without date/time prefix
        formatted_title = self._format_title(title)
        note_content.append(f"# {formatted_title}")
        if processed_content.get('tags'):
            note_content.append(' '.join(processed_content['tags']))
        note_content.append("")  # Add blank line after tags
        
        # Add source audio link
        note_content.append(f"Source Audio: ![[{audio_rel_path}]]")
        # Add main content (the transcription)
        note_content.append(processed_content.get('content', ''))
        
        try:
            note_path.write_text('\n'.join(note_content), encoding='utf-8')
            logging.info(f"Note created at {note_path}")
        except Exception as e:
            logging.error(f"Failed to create note: {str(e)}")
            raise Exception(f"Failed to create note: {str(e)}")
