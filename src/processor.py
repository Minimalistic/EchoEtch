import os
import json
import requests
import logging
import re
import time
from typing import Dict, Optional
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
        self.max_retries = 3
        self.base_delay = 1  # Base delay for exponential backoff in seconds

    def clean_text(self, text: str) -> str:
        """Clean text by removing problematic characters while preserving meaningful whitespace."""
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        # Normalize whitespace while preserving intentional line breaks
        text = re.sub(r'[\r\f\v]+', '\n', text)
        # Remove zero-width characters and other invisible unicode
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        # Normalize unicode quotes and dashes
        text = text.replace('"', '"').replace('"', '"').replace('—', '-')
        return text

    def call_ollama_with_retry(self, prompt: str) -> Optional[Dict]:
        """Make Ollama API call with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": self.temperature
                    },
                    timeout=30  # Add timeout to prevent hanging
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, json.JSONDecodeError) as e:
                delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                logging.warning(f"Ollama API attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    logging.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.error("All Ollama API attempts failed")
                    raise

    def process_transcription(self, transcription: str, audio_filename: str) -> Dict:
        """Process the transcription to generate tags and a title."""
        try:
            logging.info("Starting Ollama processing...")
            
            # Clean the transcription before processing
            cleaned_transcription = self.clean_text(transcription)
            if cleaned_transcription != transcription:
                logging.info("Cleaned problematic characters from transcription")
                
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
Transcription: {cleaned_transcription}

You must ONLY use tags from the ALLOWED TAGS list above. Do not create new tags.
If no tags from the allowed list are relevant, return an empty list.

Respond in this exact JSON format:
{{
    "title": "clear-descriptive-filename",
    "tags": ["#tag1", "#tag2"],
    "formatted_content": "The formatted transcription with single line breaks"
}}"""

            # Make API call with retry
            api_response = self.call_ollama_with_retry(prompt)
            if not api_response:
                raise ValueError("Failed to get valid response from Ollama")

            result = api_response['response']
            try:
                # Try to find the JSON object in the response using a more robust method
                json_match = re.search(r'({[\s\S]*})', result)
                if json_match:
                    result = json_match.group(1)
                
                # Clean the result string
                result = self.clean_text(result)
                
                parsed = json.loads(result)
                
                # Ensure the formatted content uses proper line endings
                if "formatted_content" in parsed:
                    content = parsed["formatted_content"]
                    # Clean the formatted content
                    content = self.clean_text(content)
                    # Replace multiple newlines with a single newline
                    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
                    parsed["formatted_content"] = content
                
                # Filter tags and return the result
                filtered_tags = self.tag_manager.filter_tags(parsed["tags"])
                return {
                    "title": parsed["title"].strip(),
                    "tags": filtered_tags,
                    "content": parsed.get("formatted_content", cleaned_transcription)
                }
            except (json.JSONDecodeError, AttributeError) as e:
                logging.error(f"Failed to parse Ollama response: {str(e)}")
                logging.debug(f"Raw response: {result}")
                # Log problematic characters if any
                if any(ord(c) < 32 and c not in '\n\t' for c in result):
                    problem_chars = [(i, ord(c)) for i, c in enumerate(result) if ord(c) < 32 and c not in '\n\t']
                    logging.debug(f"Found problematic characters at positions: {problem_chars}")
                # Fallback to using original filename and basic tags
                return {
                    "title": os.path.splitext(audio_filename)[0],
                    "tags": ["#audio", "#note"],
                    "content": cleaned_transcription
                }
                
        except Exception as e:
            logging.error(f"Ollama processing failed: {str(e)}")
            # Fallback to using original filename
            return {
                "title": os.path.splitext(audio_filename)[0],
                "tags": ["#audio", "#note"],
                "content": cleaned_transcription
            }
