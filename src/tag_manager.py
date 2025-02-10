import os
from pathlib import Path
import re
import logging

class TagManager:
    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.tags_file = vault_path / os.getenv('ALLOWED_TAGS_FILE', 'allowed_tags.md')
        logging.info(f"Looking for tags file at: {self.tags_file}")
        self._allowed_tags = set()
        self._load_tags()

    def _load_tags(self):
        """Load allowed tags from the markdown file."""
        try:
            if self.tags_file.exists():
                logging.info(f"Found tags file at {self.tags_file}")
                content = self.tags_file.read_text(encoding='utf-8')
                logging.info(f"Tags file content:\n{content}")
                # Extract tags (words starting with #) and prepend note-to-text/
                base_tags = re.findall(r'#[\w-]+', content)
                # Remove any existing #note-to-text/ prefix to avoid doubling it
                clean_tags = [tag.replace('#note-to-text/', '') for tag in base_tags]
                # Add the prefix to all tags
                self._allowed_tags = {f'#note-to-text/{tag.lstrip("#")}' for tag in clean_tags}
                logging.info(f"Loaded allowed tags: {self._allowed_tags}")
            else:
                logging.warning(f"Tags file not found at {self.tags_file}")
                self._allowed_tags = set()
        except Exception as e:
            logging.error(f"Error loading tags: {str(e)}")
            self._allowed_tags = set()

    def filter_tags(self, proposed_tags: list) -> list:
        """Filter a list of tags to only include allowed tags."""
        logging.info(f"Filtering proposed tags: {proposed_tags}")
        logging.info(f"Against allowed tags: {self._allowed_tags}")
        
        # If no allowed tags are defined, return all proposed tags with prefix
        if not self._allowed_tags:
            logging.info("No allowed tags defined, allowing all proposed tags with prefix")
            return [f'#note-to-text/{tag.lstrip("#")}' for tag in proposed_tags]
            
        # Ensure all tags start with #note-to-text/
        formatted_tags = [
            f'#note-to-text/{tag.lstrip("#")}' if not tag.startswith('#note-to-text/') else tag 
            for tag in proposed_tags
        ]
        logging.info(f"Formatted tags: {formatted_tags}")
        
        # Filter to only allowed tags
        filtered_tags = [tag for tag in formatted_tags if tag in self._allowed_tags]
        logging.info(f"After filtering: {filtered_tags}")
        
        if len(filtered_tags) < len(proposed_tags):
            logging.info(f"Filtered out {len(proposed_tags) - len(filtered_tags)} unauthorized tags")
            
        return filtered_tags
