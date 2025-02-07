import os
import json
import requests
from typing import Dict

class OllamaProcessor:
    def __init__(self):
        self.api_url = os.getenv('OLLAMA_API_URL')
        self.model = os.getenv('OLLAMA_MODEL')
        if not self.model:
            raise ValueError("OLLAMA_MODEL must be set in environment variables")
        self.temperature = float(os.getenv('OLLAMA_TEMPERATURE', '0.3'))

    def process_transcription(self, text: str) -> Dict:
        """
        Process transcription with Ollama to create structured note content
        
        Args:
            text (str): Transcribed text
            
        Returns:
            Dict: Processed content with title, content, tags, and todos
        """
        prompt = f'''
        Transform this transcription into a well-structured markdown note. You are an expert at converting
        spoken language into properly formatted written content.

        Key Processing Rules:
        1. Text Formatting:
           - Convert spoken numbers ("one", "two") to numerals in lists
           - Convert spoken "hashtag" to actual hashtag symbol (#)
           - Remove filler words and speech artifacts
           - Convert spoken punctuation into actual punctuation
        
        2. Structure Recognition:
           - Identify and properly format lists (both spoken numbers and bullet points)
           - Recognize section breaks from context and speech patterns
           - Create logical paragraph breaks
           - Add appropriate headers for different sections
        
        3. Content Enhancement:
           - Extract action items into a "Tasks" section
           - Convert spoken dates into properly formatted dates
           - Format names and technical terms appropriately
           - Identify important callouts or warnings
        
        4. Metadata Processing:
           - Extract meaningful tags from both explicit hashtags and content themes
           - Create clear, concise title from main topic
           - Only include sections that add value
           - Preserve all important information while improving readability

        Example Formatting:
        "one create documentation two test system three deploy" should become:
        1. Create documentation
        2. Test system
        3. Deploy

        "hashtag project management" should become #project-management
        
        Transcription:
        {text}
        
        Return the response in this JSON format:
        {{
            "title": "Clear, complete title (not truncated)",
            "content": "Properly formatted markdown content with appropriate sections and structure",
            "tags": ["extracted_from_hashtags_and_content"],
            "todos": ["clearly_identified_tasks"]
        }}
        '''

        try:
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
