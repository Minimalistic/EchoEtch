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

    def _extract_datetime_from_filename(self, filename: Path) -> tuple:
        """
        Extract date and time from filename if it starts with a date pattern YYYY-MM-DD
        Returns tuple of (date_str, time_str) or (None, None) if not found
        """
        try:
            # Extract date and time pattern from filename
            datetime_match = re.match(r'(\d{4}-\d{2}-\d{2})[-_](\d{2}[-_]\d{2}(?:AM|PM)?)', filename.stem, re.IGNORECASE)
            if datetime_match:
                return datetime_match.group(1), datetime_match.group(2)
            
            # If only date is found
            date_match = re.match(r'(\d{4}-\d{2}-\d{2})', filename.stem)
            if date_match:
                return date_match.group(1), None
                
            return None, None
        except Exception as e:
            logging.error(f"Error extracting datetime from filename: {str(e)}")
            return None, None

    def _get_daily_folder(self, date_str: str = None) -> Path:
        """
        Get or create a daily folder for storing audio files
        Args:
            date_str (str): Optional date string in YYYY-MM-DD format. If None, uses current date.
        Returns:
            Path: Path to the daily folder
        """
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")
        daily_folder = self.vault_path / os.getenv('NOTES_FOLDER', 'daily_notes') / date_str / "audio"
        daily_folder.mkdir(parents=True, exist_ok=True)
        return daily_folder

    def create_note(self, processed_content: Dict, audio_file: Path):
        """
        Create a markdown note in the Obsidian vault
        
        Args:
            processed_content (Dict): Processed content from Ollama
            audio_file (Path): Path to the original audio file
        """
        # Get source date from processed content or extract from filename
        source_date = processed_content.get('source_date')
        source_time = processed_content.get('source_time')
        
        if not source_date:
            source_date, source_time = self._extract_datetime_from_filename(audio_file)
        
        daily_folder = self._get_daily_folder(source_date)
        
        # Create new audio filename with source date/time or current time
        if source_date:
            if source_time:
                date_time = f"{source_date}_{source_time}"
            else:
                now = datetime.now()
                date_time = f"{source_date}_{now.strftime('%I-%M%p')}"
        else:
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
        
        # Add title
        note_content.append(f"# {title}\n")
        
        # Add audio player right after title
        note_content.extend([
            f"![[{audio_rel_path}]]\n"
        ])
        
        # Add tags if present
        if processed_content.get('tags'):
            # Strip any existing hashtags and add a single one
            formatted_tags = [tag.lstrip('#') for tag in processed_content.get('tags', [])]
            note_content.append(f"{' '.join(['#' + tag for tag in formatted_tags])}\n")
        
        # Add metadata section if there's interesting metadata
        if any(key in processed_content for key in ['language', 'confidence_issues', 'non_speech_sections']):
            note_content.append("## Metadata")
            if processed_content.get('language'):
                note_content.append(f"- Language: {processed_content['language']}")
            if processed_content.get('confidence_issues'):
                note_content.append("- Low confidence sections noted in content with [uncertain] tags")
            if processed_content.get('non_speech_sections'):
                note_content.append("- Contains non-speech sections (marked in content)")
            note_content.append("")  # Add blank line after metadata
        
        # Add content
        note_content.append(processed_content.get('formatted_content', ''))
        
        # Write the note
        try:
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(note_content))
            logging.info(f"Note created successfully at {note_path}")
        except Exception as e:
            logging.error(f"Failed to create note: {str(e)}")
            raise Exception(f"Failed to create note: {str(e)}")
