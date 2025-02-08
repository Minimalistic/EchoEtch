import os
import json
import requests
import logging
from typing import Dict

class OllamaProcessor:
    def __init__(self):
        self.api_url = os.getenv('OLLAMA_API_URL')
        self.model = os.getenv('OLLAMA_MODEL')
        if not self.model:
            raise ValueError("OLLAMA_MODEL must be set in environment variables")
        # Higher temperature for more creative formatting
        self.temperature = float(os.getenv('OLLAMA_TEMPERATURE', '0.7'))
        # Context length in tokens (default 12k)
        self.context_length = int(os.getenv('OLLAMA_CONTEXT_LENGTH', '12000'))

    def process_transcription(self, text: str) -> Dict:
        """
        Process transcription with Ollama to create structured note content
        """
        prompt = f'''
        Transform this transcription into a well-structured note with clear sections and hierarchy, formatted in proper markdown syntax.
        You are an expert at identifying structure in spoken content and creating clean, readable markdown.

        Key Processing Rules:
        1. Section Identification:
           - Create clear markdown headings (## for main sections, ### for subsections)
           - Recognize when speaker transitions between topics
           - Group related points together
           - Identify lists and enumerations
        
        2. Content Organization:
           - Start with a clear introduction/summary section
           - Group related ideas under common themed headings
           - Format action items and tasks with checkboxes (- [ ])
           - Structure meeting notes with clear headings and bullet points
           
        3. Markdown Formatting:
           - Use proper markdown list syntax (- or 1. for numbered lists)
           - Create clear paragraph breaks with blank lines
           - Use blockquotes (>) for important callouts or quotes
           - Format technical terms or important concepts with backticks or bold
           - Use proper markdown tables if data is tabular

        4. Metadata Extraction:
           - Extract meaningful tags from content
           - Identify project names and categories
           - Note any mentioned dates or deadlines
           - Capture meeting participants or references

        Format the content using proper markdown:
        - Use ## and ### for section headings
        - Use - or 1. for list items
        - Use **text** for emphasis
        - Use `code` for technical terms
        - Use > for blockquotes
        - Use - [ ] for action items
        - Use tables, code blocks, or other markdown as appropriate
        
        Input Text:
        {text}
        
        Return ONLY a JSON response in this exact format:
        {{
            "title": "Clear, complete title (not truncated)",
            "content": "Content in proper markdown format",
            "tags": ["extracted_tags"],
            "todos": ["clearly_identified_tasks"]
        }}
        '''

        try:
            logging.info("Starting initial Mistral processing...")
            logging.debug(f"Input text length: {len(text)} characters")
            
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                    "context_length": self.context_length
                }
            )
            response.raise_for_status()
            
            result = response.json()
            processed_text = result['response']
            logging.info(f"Received Mistral response: {len(processed_text)} characters")
            logging.debug(f"Raw Mistral response: {processed_text[:200]}...")
            
            try:
                initial_result = json.loads(processed_text)
                logging.info("Successfully parsed Mistral JSON response")
                return initial_result
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse Mistral JSON: {str(e)}")
                logging.error(f"Raw response that failed to parse: {processed_text}")
                return {
                    "title": text[:30].strip() + "...",
                    "content": text,
                    "tags": [],
                    "todos": []
                }
                
        except Exception as e:
            logging.error(f"Mistral processing failed: {str(e)}")
            raise
