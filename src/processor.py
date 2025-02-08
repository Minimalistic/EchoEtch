import os
import json
import requests
import logging
import re
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

    def _send_to_ollama(self, prompt: str) -> str:
        try:
            logging.info("Starting initial Ollama processing...")
            logging.debug(f"Input text length: {len(prompt)} characters")
            
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
            
            result = response.json()['response']
            logging.info(f"Received Ollama response: {len(result)} characters")
            logging.debug(f"Raw Ollama response: {result[:500]}...")
            return result
        except Exception as e:
            logging.error(f"Ollama processing failed: {str(e)}")
            raise

    def process_transcription(self, text: str) -> Dict:
        """
        Process transcription with Ollama to create structured note content
        """
        prompt = '''
        You are an expert at converting spoken text into well-structured markdown notes.
        Convert the following transcription into markdown, following these rules:

        1. Document Structure:
           - Start with a single "# " (with a space) for the main title
           - Place all tags on the line immediately after the title, WITH # prefix (e.g., "#development #project-management")
           - Use ## for major sections
           - Use ### for subsections
           - Never repeat the title anywhere in the document
           - Maximum one empty line between sections
           
        2. Text Formatting and Content Rules:
           - Keep sentences together on the same line - don't split them across lines
           - Use **bold** for emphasis within text
           - Use *italic* for lighter emphasis
           - Use `code` for technical terms
           - For bullet points:
             * Use "- " (hyphen + space) for bullet points
             * Indent bullet points with 2 spaces
             * Keep bullet point text on the same line
           - For numbered lists:
             * Keep the entire point on one line
             * Don't split sentences across lines unnecessarily
           - Never leave empty sections (like an empty "Important:" section)
           - Remove any unnecessary line breaks within paragraphs

        3. Task Handling:
           - Create tasks only when there are clear action items:
             * Explicit statements about tasks that need to be done
             * Items specifically marked as "todo" or "to do"
             * Items marked as "Important" that indicate required actions
             * Clear follow-up actions or commitments
             * Specific deadlines that require action
           - Don't force regular statements or information into tasks
           - Don't create tasks from general discussion points or information sharing
           - Never write "Todo:" or "Important:" in the text - convert these directly to tasks
           - Never mention a task in the content if it's in the tasks section
           - Don't create empty sections

        Format your response as a JSON object with these exact fields:
        {
            "title": "The main title without any # prefix",
            "content": "The full markdown content with proper newlines escaped as \\n",
            "tags": ["tag1", "tag2"],
            "tasks": ["task1", "task2"]
        }

        Here's the transcription to process:
''' + text

        try:
            response = self._send_to_ollama(prompt)
            logging.debug(f"Processing response: {response[:500]}...")
            
            # First try to parse as JSON directly
            try:
                result = json.loads(response)
                logging.info("Successfully parsed JSON response")
                if not isinstance(result, dict):
                    raise ValueError("Response is not a dictionary")
                if 'title' not in result:
                    raise ValueError("Response missing required 'title' field")
                
                # Clean up the content
                if 'content' in result:
                    # Remove multiple empty lines and clean up content
                    lines = result['content'].split('\n')
                    cleaned_lines = []
                    prev_empty = False
                    section_header = None
                    
                    for line in lines:
                        line = line.rstrip()
                        
                        # Skip empty sections
                        if line.startswith('##'):
                            section_header = line
                            continue
                        elif section_header:
                            if line.strip():
                                cleaned_lines.append(section_header)
                                cleaned_lines.append(line)
                            section_header = None
                            continue
                            
                        # Skip lines that mention tasks
                        if any(task.lower() in line.lower() for task in result.get('tasks', [])):
                            continue
                            
                        # Handle empty lines
                        if not line.strip():
                            if not prev_empty:
                                cleaned_lines.append('')
                                prev_empty = True
                        else:
                            # Ensure bullet points are properly indented
                            if line.lstrip().startswith('- '):
                                line = '  ' + line.lstrip()
                            cleaned_lines.append(line)
                            prev_empty = False
                    
                    # Remove trailing empty lines
                    while cleaned_lines and not cleaned_lines[-1].strip():
                        cleaned_lines.pop()
                        
                    result['content'] = '\n'.join(cleaned_lines)
                
                # Ensure tags have # prefix
                if 'tags' in result:
                    result['tags'] = ['#' + tag.lstrip('#') for tag in result['tags']]
                
                return result
            except json.JSONDecodeError as e:
                logging.info(f"Direct JSON parse failed: {str(e)}")
                # If direct JSON parsing fails, try to extract JSON from the response
                # Look for content between triple backticks if present
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        logging.info("Successfully parsed JSON from code block")
                        if not isinstance(result, dict):
                            raise ValueError("Response is not a dictionary")
                        if 'title' not in result:
                            raise ValueError("Response missing required 'title' field")
                        return result
                    except json.JSONDecodeError as e:
                        logging.info(f"JSON code block parse failed: {str(e)}")
                
                # If we still don't have valid JSON, create it from the markdown
                logging.info("Falling back to markdown parsing")
                lines = response.strip().split('\n')
                title = None
                tags = []
                tasks = []
                content_lines = []
                in_tasks_section = False
                prev_empty = False

                for line in lines:
                    line = line.rstrip()
                    
                    # Handle title
                    if not title and line.strip().startswith('# '):
                        title = line[2:].strip()
                        continue
                        
                    # Handle tags
                    if not tags and not line.startswith('#') and ' ' in line:
                        potential_tags = line.strip().split()
                        if all(tag.isalnum() or '-' in tag for tag in potential_tags):
                            tags = potential_tags
                            continue
                    
                    # Handle tasks section
                    if line.strip().lower() == '## tasks':
                        in_tasks_section = True
                        continue
                        
                    # Collect tasks
                    if line.strip().startswith('- [ ]'):
                        tasks.append(line[6:].strip())
                        continue
                        
                    # Handle regular content
                    if not in_tasks_section:
                        # Skip empty "Important:" sections
                        if line.strip().lower() == "important:":
                            # Look ahead to see if the section is empty
                            next_non_empty = next((l for l in lines[lines.index(line)+1:] if l.strip()), "").strip()
                            if not next_non_empty or next_non_empty.startswith("##"):
                                continue
                            
                        # Manage empty lines
                        if not line.strip():
                            if not prev_empty:
                                content_lines.append('')
                                prev_empty = True
                        else:
                            content_lines.append(line)
                            prev_empty = False

                if not title:
                    logging.warning("No title found in markdown, using default")
                    title = "Untitled Note"

                # Create proper JSON response
                result = {
                    "title": title,
                    "content": '\n'.join(content_lines).strip(),
                    "tags": tags,
                    "tasks": tasks
                }
                logging.info("Successfully created result from markdown")
                return result

        except Exception as e:
            logging.error(f"Error during Ollama processing: {str(e)}")
            return {
                "title": "Untitled Note",
                "content": text,
                "tags": [],
                "tasks": []
            }
