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
        1. Document Structure:
           - Use # for the main title
           - Use ## for major sections
           - Use ### for subsections
           - Use #### for deeper subsections
           - Avoid using bold text for section headers - use appropriate heading levels instead
           
        2. Text Formatting:
           - Use **bold** only for emphasis within text, not for structural elements
           - Use *italic* for lighter emphasis or terms being defined
           - Use `code` for technical terms, commands, or file names
           - Use > only for actual quotes or important callouts
           - Use --- for horizontal rules to separate major content areas if needed
        
        3. Lists and Items:
           - Use - for unordered lists
           - Use 1. for ordered lists when sequence matters
           - Use - [ ] for action items/todos
           - Maintain consistent indentation for nested lists
           - Use regular bullet points for feature lists, not blockquotes
           
        4. Content Organization:
           - Start with a clear title and introduction
           - Group related content under appropriate heading levels
           - Use lists for multiple related items
           - Create clear paragraph breaks with blank lines
           - Use tables for structured data
           
        5. Metadata and Special Sections:
           - Create a ## Tags section with hashtags
           - Create a ## Todos section with checkboxes for ALL action items
           - Add a ## Source section if applicable
           - Format dates as YYYY-MM-DD
           - Use consistent formatting for links and references

        6. Todo Extraction (Important):
           - Extract ONLY concrete, actionable tasks
           - Look for tasks that have:
             * Clear next actions (e.g., "talk to Mike about X", "update the docs")
             * Specific deadlines or timeframes
             * Defined deliverables or outcomes
           - Do NOT convert to todos:
             * Rhetorical statements ("I need to figure out why...")
             * General musings or wonderings
             * Personal reflections or questions
             * Vague or undefined challenges
           - When in doubt, prefer NOT creating a todo unless it's clearly actionable
           - Add identified todos to both the content's Todo section AND the todos array in the JSON response

        7. Voice Commands:
           Always recognize these explicit voice commands:
           - "Make a todo: <task>" -> Create a todo item
           - "Add todo: <task>" -> Create a todo item
           - "Create section: <title>" -> Create a new section with given title
           - "New section: <title>" -> Create a new section with given title
           - "Tag this: <tag>" -> Add a tag
           - "Add tag: <tag>" -> Add a tag
           - "Quote this: <text>" -> Format as blockquote
           - "Important note: <text>" -> Format as callout or emphasis
           - "Technical note: <text>" -> Format as technical note with code formatting
           These commands take precedence over regular text interpretation and should always be processed literally.

        Input Text:
        {text}
        
        Return ONLY a JSON response. IMPORTANT:
        1. Use double quotes for all JSON keys and string values
        2. Properly escape all newlines in the content as \\n
        3. Do not use any actual newlines or indentation in the JSON content
        4. Format exactly like this example:
        {{"title":"Example Title","content":"# Heading\\n\\n## Section\\n- Point 1\\n- Point 2\\n\\n## Todos\\n- [ ] Task 1\\n- [ ] Task 2","tags":["tag1","tag2"],"todos":["Task 1","Task 2"]}}
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
            
            # Clean up the response to handle markdown code blocks and other formatting
            processed_text = processed_text.strip()
            # Remove markdown code block markers if present
            if processed_text.startswith('```'):
                processed_text = '\n'.join(processed_text.split('\n')[1:-1])
            # Remove any "json" language identifier if present
            processed_text = processed_text.replace('json\n', '', 1)
            # Ensure we're working with clean JSON by removing any leading/trailing whitespace
            processed_text = processed_text.strip()
            
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
