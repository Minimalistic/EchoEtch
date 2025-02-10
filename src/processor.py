import os
import json
import requests
import logging
from typing import Dict
from pathlib import Path
from .tag_manager import TagManager

class OllamaProcessor:
    def __init__(self):
        self.api_url = os.getenv('OLLAMA_API_URL')
        self.model = os.getenv('OLLAMA_MODEL')
        if not self.model:
            raise ValueError("OLLAMA_MODEL must be set in environment variables")
        self.temperature = float(os.getenv('OLLAMA_TEMPERATURE', '0.3'))
        vault_path = Path(os.getenv('OBSIDIAN_VAULT_PATH'))
        self.tag_manager = TagManager(vault_path)

    def process_transcription(self, transcription: str, audio_filename: str) -> Dict:
        """Process the transcription to generate tags and a title."""
        try:
            logging.info("Starting Ollama processing...")
            
            # Get the list of allowed tags
            allowed_tags = list(self.tag_manager._allowed_tags)
            allowed_tags_str = '\n'.join(allowed_tags)
            
            prompt = f"""Given this transcription from an audio note and the list of ALLOWED TAGS below, generate:
1. A clear, concise filename (without extension)
2. The most relevant tags from ONLY the allowed tags list that apply to this content

ALLOWED TAGS:
{allowed_tags_str}

Original audio filename: {audio_filename}
Transcription: {transcription}

You must ONLY use tags from the ALLOWED TAGS list above. Do not create new tags.
If no tags from the allowed list are relevant, return an empty list.

Respond in this exact JSON format:
{{
    "title": "clear-descriptive-filename",
    "tags": ["#tag1", "#tag2"]  // Only use tags from the allowed list
}}"""

            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature
                }
            )
            response.raise_for_status()
            
            result = response.json()['response']
            try:
                parsed = json.loads(result)
                # We still filter just to be safe, but the AI should now only use allowed tags
                filtered_tags = self.tag_manager.filter_tags(parsed["tags"])
                return {
                    "title": parsed["title"].strip(),
                    "tags": filtered_tags,
                    "content": transcription  # Use original transcription as content
                }
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse Ollama response: {str(e)}")
                # Fallback to using original filename and basic tags
                return {
                    "title": os.path.splitext(audio_filename)[0],
                    "tags": ["#audio", "#note"],
                    "content": transcription
                }
                
        except Exception as e:
            logging.error(f"Ollama processing failed: {str(e)}")
            # Fallback to using original filename
            return {
                "title": os.path.splitext(audio_filename)[0],
                "tags": ["#audio", "#note"],
                "content": transcription
            }
