import os
import json
import requests
from typing import Dict

class OllamaProcessor:
    def __init__(self):
        self.api_url = os.getenv('OLLAMA_API_URL')
        self.model = os.getenv('OLLAMA_MODEL', 'mistral')

    def process_transcription(self, text: str) -> Dict:
        """
        Process transcription with Ollama to create structured note content
        
        Args:
            text (str): Transcribed text
            
        Returns:
            Dict: Processed content with title, content, tags, and todos
        """
        prompt = f"""
        Process this transcription into a well-formatted markdown note.
        Extract key information, add relevant tags, and identify any todo items.
        Make sure to include the full transcription in the content section.
        
        Transcription:
        {text}
        
        Return the response in this JSON format:
        {{
            "title": "Generated title",
            "content": "## Transcription\\n\\n[Full transcription text here]\\n\\n## Notes\\n\\n[Any additional insights or key points]",
            "tags": ["tag1", "tag2"],
            "todos": ["todo item 1", "todo item 2"]
        }}
        
        The content MUST include the full transcription text.
        """

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            
            # Extract JSON from Ollama's response
            result = response.json()
            processed_text = result['response']
            
            # Parse the JSON response
            try:
                return json.loads(processed_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "title": "Note " + text[:30].strip() + "...",
                    "content": text,
                    "tags": [],
                    "todos": []
                }
                
        except Exception as e:
            raise Exception(f"Ollama processing failed: {str(e)}")
