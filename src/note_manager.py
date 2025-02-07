import os
from datetime import datetime
from pathlib import Path
from typing import Dict
import re
import logging
import subprocess
import platform
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
        logging.debug(f"Daily folder path: {daily_folder}")
        logging.debug(f"Daily folder exists: {daily_folder.exists()}")
        daily_folder.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Daily folder created/verified")
        return daily_folder

    def create_note(self, processed_content: Dict, audio_file: Path):
        """
        Create a markdown note in the Obsidian vault
        
        Args:
            processed_content (Dict): Processed content from Ollama
            audio_file (Path): Path to the original audio file
        """
        import subprocess
        import platform
        
        # First, handle the audio file move
        daily_folder = self._get_daily_folder()
        new_audio_path = daily_folder / audio_file.name
        
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
                # Use shutil for file operations
                import shutil
                
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
        
        # Now create the note
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        sanitized_title = self._sanitize_filename(processed_content['title'][:30])
        note_filename = f"{timestamp}_{sanitized_title}.md"
        note_path = self.notes_folder / note_filename
        
        # Create relative link to audio file in daily folder
        audio_rel_path = os.path.relpath(new_audio_path, self.notes_folder)
        
        # Build note content
        note_content = [f"# {processed_content['title']}\n"]
        note_content.append(processed_content['content'])
        
        if processed_content.get('tags') and len(processed_content['tags']) > 0:
            note_content.append("\n## Tags")
            note_content.append(' '.join(['#' + tag for tag in processed_content['tags']]))
        
        if processed_content.get('todos') and len(processed_content['todos']) > 0:
            note_content.append("\n## Todos")
            note_content.append('\n'.join(['- [ ] ' + todo for todo in processed_content['todos']]))
        
        note_content.append(f"\n## Source\n- [[{audio_rel_path}|Original Audio]]")
        
        try:
            note_path.write_text('\n'.join(note_content), encoding='utf-8')
            logging.info(f"Note created at {note_path}")
        except Exception as e:
            logging.error(f"Failed to create note: {str(e)}")
            raise Exception(f"Failed to create note: {str(e)}")
