import os
import json
import requests
import logging
from typing import Dict

class OllamaProcessor:
    def __init__(self):
        self.api_url = os.getenv('OLLAMA_API_URL')
        self.model = os.getenv('OLLAMA_MODEL')
        self.gemma_model = os.getenv('GEMMA_MODEL')
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
        Transform this transcription into a well-structured note with clear sections and hierarchy.
        You are an expert at identifying structure in spoken content.

        Key Processing Rules:
        1. Section Identification:
           - Identify main topics and create sections
           - Recognize when speaker transitions between topics
           - Group related points together
           - Identify lists and enumerations
        
        2. Content Organization:
           - Create a clear introduction/summary section
           - Group related ideas under common themes
           - Identify action items and tasks
           - Detect meeting notes or discussion points
           
        3. Text Structure:
           - Convert spoken lists into proper list format
           - Create clear paragraph breaks
           - Identify items that should be callouts or quotes
           - Mark technical terms or important concepts

        4. Metadata Extraction:
           - Extract meaningful tags from content
           - Identify project names and categories
           - Note any mentioned dates or deadlines
           - Capture meeting participants or references

        Structure the content with these markers for the formatting pass:
        - Use [SECTION] to mark main sections
        - Use [LIST] to mark list items
        - Use [IMPORTANT] for critical points
        - Use [TECH] for technical terms
        - Use [QUOTE] for quotations or references
        - Use [ACTION] for action items
        - Use [MEETING] for meeting notes
        
        Input Text:
        {text}
        
        Return ONLY a JSON response in this exact format:
        {{
            "title": "Clear, complete title (not truncated)",
            "content": "Content with structural markers",
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
                return self._refine_with_gemma(initial_result)
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

    def _refine_with_gemma(self, note_data: Dict) -> Dict:
        """
        Apply markdown formatting to the structured content.
        """
        prompt = f'''
        You are a markdown formatting expert. Transform this pre-structured note into beautifully formatted markdown
        while preserving all content and meaning. The input contains structural markers that you should convert to proper markdown.

        Formatting Rules:
        1. Convert markers to markdown:
           - [SECTION] → # (main header)
           - [LIST] → Proper markdown lists (- or 1.)
           - [IMPORTANT] → > blockquote
           - [TECH] → `code` or **bold**
           - [QUOTE] → > blockquote
           - [ACTION] → - [ ] task
           - [MEETING] → ## Meeting Notes

        2. Additional Formatting:
           - Add proper heading hierarchy (# → ##)
           - Ensure consistent list formatting
           - Add horizontal rules (---) between major sections
           - Format dates and times consistently
           - Create proper markdown links
           - Use emphasis (*italic*) for subtle highlights
           - Add bullet points for unstructured lists
           - Create tables if data is tabular

        3. Final Structure:
           - Start with a # Title
           - Add a brief summary if present
           - Organize content with clear headings
           - End with metadata (tags, references)
           - Ensure proper spacing between sections

        Current note content to format:
        {note_data['content']}
        
        Return ONLY a JSON response in this exact format:
        {{
            "title": "{note_data['title']}",
            "content": "Properly formatted markdown content",
            "tags": {json.dumps(note_data['tags'])},
            "todos": {json.dumps(note_data['todos'])}
        }}
        '''

        try:
            logging.info("Starting Gemma formatting pass...")
            response = requests.post(
                self.api_url,
                json={
                    "model": self.gemma_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                    "context_length": self.context_length
                }
            )
            response.raise_for_status()
            
            result = response.json()
            refined_text = result['response']
            logging.info(f"Received Gemma response: {len(refined_text)} characters")
            logging.debug(f"Raw Gemma response: {refined_text[:200]}...")
            
            try:
                final_result = json.loads(refined_text)
                logging.info("Successfully parsed Gemma JSON response")
                return final_result
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse Gemma JSON: {str(e)}")
                logging.error(f"Raw response that failed to parse: {refined_text}")
                return note_data
                
        except Exception as e:
            logging.error(f"Gemma formatting failed: {str(e)}")
            return note_data
