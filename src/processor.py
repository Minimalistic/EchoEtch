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

    def process_transcription(self, transcription_data: dict, audio_filename: str) -> Dict:
        """Process the transcription to generate tags and a title."""
        try:
            logging.info("Starting Ollama processing...")
            
            # Clean the transcription before processing
            cleaned_transcription = self.clean_text(transcription_data["text"])
            
            # Get the list of allowed tags
            allowed_tags = list(self.tag_manager._allowed_tags)
            allowed_tags_str = '\n'.join(allowed_tags)
            
            # Build metadata information
            metadata = []
            
            # Add language information
            if transcription_data.get("language"):
                metadata.append(f"Language: {transcription_data['language']}")
            
            # Process segments to identify notable features
            segments_info = []
            for segment in transcription_data.get("segments", []):
                # Calculate pause before this segment
                if segments_info:  # Not the first segment
                    pause_duration = segment["start"] - segments_info[-1]["end"]
                    if pause_duration > 1.0:  # Only note significant pauses
                        segments_info.append({
                            "type": "pause",
                            "duration": round(pause_duration, 1),
                            "position": segment["start"]
                        })
                
                # Add segment with confidence info
                segments_info.append({
                    "type": "segment",
                    "start": segment["start"],
                    "end": segment["end"],
                    "confidence": segment["confidence"],
                    "no_speech_prob": segment["no_speech_prob"]
                })
            
            # Convert segments info to readable format
            for info in segments_info:
                if info["type"] == "pause":
                    metadata.append(f"[Pause: {info['duration']}s at {info['position']}s]")
                elif info["type"] == "segment":
                    if info["no_speech_prob"] > 0.5:  # Likely background noise or non-speech
                        metadata.append(f"[Non-speech section: {info['start']}-{info['end']}s]")
                    elif info["confidence"] < -1.0:  # Low confidence section
                        metadata.append(f"[Low confidence section: {info['start']}-{info['end']}s]")
            
            metadata_str = "\n".join(metadata)
            
            prompt = f"""Given this transcription from an audio note with metadata and the list of ALLOWED TAGS below:

METADATA:
{metadata_str}

1. Format the transcription by:
   - Using the metadata to inform natural breaks and section divisions
   - Adding appropriate line breaks where pauses or topic changes occur
   - Using markdown headers (# or ##) for major topic changes or sections
   - Adding [uncertain] tags for low confidence sections
   - Preserving the original meaning and content
   - Using single line breaks between paragraphs
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

            response = self.call_ollama_with_retry(prompt)
            if not response or "response" not in response:
                raise ValueError("Invalid response from Ollama")

            try:
                result = json.loads(response["response"])
                # Validate the result has required fields
                if not all(k in result for k in ["title", "tags", "formatted_content"]):
                    raise ValueError("Missing required fields in Ollama response")
                return result
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in Ollama response")
                
        except Exception as e:
            logging.error(f"Error processing transcription: {str(e)}")
            raise
