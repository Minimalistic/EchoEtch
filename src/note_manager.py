import os
from datetime import datetime
from pathlib import Path
from typing import Dict
import re

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

    def create_note(self, processed_content: Dict, audio_file: Path):
        """
        Create a markdown note in the Obsidian vault
        
        Args:
            processed_content (Dict): Processed content from Ollama
            audio_file (Path): Path to the original audio file
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        sanitized_title = self._sanitize_filename(processed_content['title'][:30])
        note_filename = f"{timestamp}_{sanitized_title}.md"
        note_path = self.notes_folder / note_filename

        # Create relative link to audio file
        audio_rel_path = os.path.relpath(audio_file, self.notes_folder)
        
        # Build note content
        note_content = f"""# {processed_content['title']}

{processed_content['content']}

## Tags
{' '.join(['#' + tag for tag in processed_content['tags']])}

## Todos
{chr(10).join(['- [ ] ' + todo for todo in processed_content['todos']])}

## Source
- [[{audio_rel_path}|Original Audio]]
"""

        try:
            # Write the note
            note_path.write_text(note_content, encoding='utf-8')
            
            # Move audio file to Obsidian vault if it's not already there
            if not audio_file.is_relative_to(self.vault_path):
                new_audio_path = self.notes_folder / audio_file.name
                audio_file.rename(new_audio_path)
                
        except Exception as e:
            raise Exception(f"Failed to create note: {str(e)}")
