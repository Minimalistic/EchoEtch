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
        
        # Now create the note with matching naming convention
        note_filename = f"{date_time}_{sanitized_title}.md"
        # Place the note in the same day folder as the audio, but one level up
        note_path = daily_folder.parent / note_filename
        
        # Create relative link to audio file
        # Use relative path from vault root to audio file for Obsidian compatibility
        audio_rel_path = os.path.relpath(new_audio_path, self.vault_path)
        audio_rel_path = audio_rel_path.replace('\\', '/')  # Convert Windows path to forward slashes for markdown
        
        # Build note content
        note_content = []
        
        # Add title and tags at the top
        note_content.append(f"# {title}")
        if processed_content.get('tags') and len(processed_content['tags']) > 0:
            # Tags should already have # prefix from processor
            note_content.append(' '.join(processed_content['tags']))
        note_content.append("")  # Add blank line after tags
        
        # Add main content
        if isinstance(processed_content.get('content'), list):
            content = '\n'.join(processed_content['content'])
        else:
            content = str(processed_content.get('content', ''))
            
        # Clean up the content
        content_lines = content.split('\n')
        filtered_lines = []
        prev_empty = False
        skip_next_line = False
        section_header = None
        
        for i, line in enumerate(content_lines):
            line = line.rstrip()  # Remove trailing whitespace
            
            # Skip empty sections
            if line.startswith('##'):
                section_header = line
                continue
            elif section_header:
                if line.strip():
                    filtered_lines.append(section_header)
                    filtered_lines.append(line)
                section_header = None
                continue
            
            # Skip the title line and any duplicate tag lines
            if line.startswith('# ') and line[2:].strip() == title:
                continue
            if any(tag.lstrip('#') in line for tag in processed_content.get('tags', [])):
                continue
            if line.strip() == '## Tags':
                skip_next_line = True
                continue
            if skip_next_line:
                skip_next_line = False
                continue
                
            # Skip lines that mention tasks
            if processed_content.get('tasks'):
                if any(task.lower() in line.lower() for task in processed_content['tasks']):
                    continue
            
            # Handle bullet points
            if line.lstrip().startswith('- '):
                line = '  ' + line.lstrip()  # Ensure consistent indentation
                
            # Handle empty lines - maximum one empty line between content
            if not line.strip():
                if not prev_empty:
                    filtered_lines.append('')
                    prev_empty = True
            else:
                filtered_lines.append(line)
                prev_empty = False
        
        # Remove any trailing empty lines before adding to note_content
        while filtered_lines and not filtered_lines[-1].strip():
            filtered_lines.pop()
            
        # Remove any empty sections
        i = 0
        while i < len(filtered_lines):
            if filtered_lines[i].startswith('##'):
                if i + 1 >= len(filtered_lines) or not filtered_lines[i + 1].strip():
                    filtered_lines.pop(i)
                    continue
            i += 1
            
        note_content.extend(filtered_lines)
        
        # Add tasks section if present
        if processed_content.get('tasks') and len(processed_content['tasks']) > 0:
            if note_content and note_content[-1].strip():  # If last line isn't empty
                note_content.append("")  # Add single empty line before tasks
            note_content.append("## Tasks")
            note_content.append('\n'.join(['- [ ] ' + task for task in processed_content['tasks']]))
        
        # Add source section
        if note_content and note_content[-1].strip():  # If last line isn't empty
            note_content.append("")  # Add single empty line before source
        note_content.append("## Source")
        note_content.append(f"![[{audio_rel_path}|Original Audio]]")
        
        # Add original transcription in collapsible section
        note_content.append("")  # Add empty line for spacing
        note_content.append("### Original Transcription")
        note_content.append("> [!abstract]- Click to view original transcription")
        note_content.append("> ```")
        note_content.append("> " + processed_content.get('original_transcription', 'Original transcription not available').replace('\n', '\n> '))
        note_content.append("> ```")
        
        try:
            note_path.write_text('\n'.join(note_content), encoding='utf-8')
            logging.info(f"Note created at {note_path}")
        except Exception as e:
            logging.error(f"Failed to create note: {str(e)}")
            raise Exception(f"Failed to create note: {str(e)}")
