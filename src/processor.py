import os
import json
import requests
import logging
import re
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
            
            prompt = f"""Given this transcription from an audio note and the list of ALLOWED TAGS below:
1. Format the transcription by:
   - Adding appropriate line breaks where natural pauses or topic changes occur
   - Using only single line breaks between paragraphs (no multiple blank lines)
   - Adding markdown headers (# or ##) for major topic changes or sections, but only when it clearly makes sense
   - DO NOT prefix headers with "Note:" - use concise, descriptive headers that reflect the content
   - Preserving the original meaning and content
2. Generate a clear, concise filename (without extension)
3. Select the most relevant tags from ONLY the allowed tags list that apply to this content

IMPORTANT: Your response must be valid JSON. Escape any special characters in the formatted content.

ALLOWED TAGS:
{allowed_tags_str}

Original audio filename: {audio_filename}
Transcription: {transcription}

You must ONLY use tags from the ALLOWED TAGS list above. Do not create new tags.
If no tags from the allowed list are relevant, return an empty list.

Respond in this exact JSON format:
{{
    "title": "clear-descriptive-filename",
    "tags": ["#tag1", "#tag2"],
    "formatted_content": "The formatted transcription with single line breaks"
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
                # Try to find the JSON object in the response using a more robust method
                json_match = re.search(r'({[\s\S]*})', result)
                if json_match:
                    result = json_match.group(1)
                
                # Clean the result string - remove any invalid control characters
                result = ''.join(char for char in result if ord(char) >= 32 or char in '\n\r\t')
                
                parsed = json.loads(result)
                
                # Ensure the formatted content uses proper line endings
                if "formatted_content" in parsed:
                    # Replace multiple newlines with a single newline
                    parsed["formatted_content"] = re.sub(r'\n\s*\n\s*\n+', '\n\n', parsed["formatted_content"])
                    parsed["formatted_content"] = parsed["formatted_content"].replace('\\n', '\n')
                
                # Filter tags and return the result
                filtered_tags = self.tag_manager.filter_tags(parsed["tags"])
                return {
                    "title": parsed["title"].strip(),
                    "tags": filtered_tags,
                    "content": parsed.get("formatted_content", transcription)
                }
            except (json.JSONDecodeError, AttributeError) as e:
                logging.error(f"Failed to parse Ollama response: {str(e)}")
                logging.debug(f"Raw response: {result}")
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
