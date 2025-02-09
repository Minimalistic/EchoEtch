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
        # Lower temperature for more consistent formatting
        self.temperature = float(os.getenv('OLLAMA_TEMPERATURE', '0.3'))
        # Context length in tokens (default 12k)
        self.context_length = int(os.getenv('OLLAMA_CONTEXT_LENGTH', '12000'))

    def _send_to_ollama(self, prompt: str, temperature: float = None) -> str:
        try:
            logging.info("Starting Ollama processing...")
            logging.debug(f"Input text length: {len(prompt)} characters")
            
            # Use provided temperature or fall back to default
            temp = temperature if temperature is not None else self.temperature
            
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temp,
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

    def _validate_and_parse_response(self, response: str) -> Dict:
        """Validate and parse the LLM response into a dictionary"""
        try:
            # First try to parse as JSON directly
            content = json.loads(response)
            
            # Validate required fields
            required_fields = ["title", "content", "tags", "tasks"]
            if not all(field in content for field in required_fields):
                raise ValueError("Missing required fields in response")
                
            return content
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Invalid response format: {str(e)}")
            raise

    def _evaluate_conversion(self, original_text: str, converted_content: Dict) -> Dict:
        """Have the LLM evaluate the quality of its conversion"""
        evaluation_prompt = f'''
            You are a quality assurance expert. Evaluate how well the following markdown note captures and structures the original transcription.
            Rate the conversion on these aspects (1-10 scale):
            1. Completeness: Are all key points from the original included?
            2. Structure: Is the content well-organized with appropriate sections and formatting?
            3. Clarity: Is the converted text clear and readable?
            4. Task Handling: Are tasks (if any) properly extracted and formatted?
            5. Overall Quality: Overall rating considering all factors

            Provide your response as a JSON object with numerical scores and a brief explanation:
            {{
                "completeness": 8,
                "structure": 9,
                "clarity": 8,
                "task_handling": 9,
                "overall_quality": 8.5,
                "explanation": "Brief explanation of the ratings"
            }}

            Original Transcription:
            {original_text}

            Converted Note:
            # {converted_content['title']}
            {' '.join(converted_content['tags'])}
            
            {converted_content['content']}
            
            Tasks:
            {chr(10).join(['- [ ] ' + task for task in converted_content['tasks']])}
        '''

        try:
            response = self._send_to_ollama(evaluation_prompt, temperature=0.2)  # Lower temperature for more consistent evaluation
            evaluation = json.loads(response)
            logging.info(f"Conversion quality evaluation: {evaluation['overall_quality']}/10")
            return evaluation
        except Exception as e:
            logging.warning(f"Failed to evaluate conversion: {str(e)}")
            return {
                "completeness": 0,
                "structure": 0,
                "clarity": 0,
                "task_handling": 0,
                "overall_quality": 0,
                "explanation": "Evaluation failed"
            }

    def _compare_responses(self, responses: list[Dict]) -> Dict:
        """Compare multiple responses and select the best one"""
        if not responses:
            raise ValueError("No valid responses to compare")
        
        def score_response(response: Dict) -> float:
            score = 0.0
            
            # Score based on title quality (non-empty, reasonable length)
            if response["title"] and 3 <= len(response["title"].split()) <= 10:
                score += 1.0
                
            # Score based on content structure
            content = response["content"]
            if "##" in content:  # Has sections
                score += 1.0
            if "###" in content:  # Has subsections
                score += 0.5
            if "- " in content or "1. " in content:  # Has lists
                score += 0.5
                
            # Score based on markdown formatting usage
            if any(marker in content for marker in ["**", "*", "`"]):
                score += 0.5
                
            # Score based on tags (prefer 2-5 tags)
            num_tags = len(response["tags"])
            if 2 <= num_tags <= 5:
                score += 1.0
            
            # Penalize extremely long or short content
            content_length = len(content)
            if 100 <= content_length <= 5000:
                score += 1.0
            elif content_length > 5000:
                score -= 0.5
            
            # Add the LLM's self-evaluation score
            if 'conversion_quality' in response:
                score += response['conversion_quality']['overall_quality'] / 2  # Scale down to match other scores
                
            return score
        
        # Score each response and return the best one
        scored_responses = [(score_response(r), r) for r in responses]
        best_response = max(scored_responses, key=lambda x: x[0])[1]
        
        logging.info(f"Selected best response with title: {best_response['title']}")
        return best_response

    def process_transcription(self, text: str) -> Dict:
        """
        Process transcription with multiple Ollama passes to create structured note content
        """
        # Add repetition detection
        def detect_repetition(text: str) -> str:
            # Split into sentences and remove duplicates while preserving order
            sentences = text.split('. ')
            seen = set()
            unique_sentences = []
            for sentence in sentences:
                if sentence.strip() and sentence not in seen:
                    seen.add(sentence)
                    unique_sentences.append(sentence)
            return '. '.join(unique_sentences)

        # Clean the input text before processing
        text = detect_repetition(text)
        
        prompt = '''
            You are an expert at converting transcribed text into Obsidian-compatible markdown notes. Convert the transcription below into a well-structured markdown note following these guidelines:

            1. Title & Tags:
            - Start with a main title using "# " (the JSON "title" field should not include the "#").
            - On the next line, include tags where each tag starts with "#", if there is more than one word, separate each tag with a "-".

            2. Content Structure:
            - Use "##" for major sections and "###" for subsections.
            - Include all important details from the transcription without omitting or inferring extra content.
            - Keep complete sentences on one line and remove unnecessary line breaks.

            3. Text Formatting:
            - Apply **bold**, *italic*, and `code` formatting as appropriate.
            - Use "- " for bullet lists and "1. " for numbered lists, if a list is nested in another, tab it.

            4. Task Extraction (IMPORTANT):
            - **Only** extract tasks if the transcription explicitly includes the exact phrases "Make Todo:" or "Todo:".
            - Do not infer tasks from content that merely resembles a task.
            - When a task marker is present, remove the marker from the final text.
            - Format each task as "- [ ] Task description" and list them under a "## Tasks" section.
            - If no explicit task markers are present, do not include any tasks (the tasks array must be empty).

            Format your response as a JSON object exactly in this format:
            {
                "title": "Title text (without leading '# ')",
                "content": "Full markdown content with proper newlines escaped as \\n",
                "tags": ["#tag1", "#tag2"],
                "tasks": ["task description 1", "task description 2"]
            }

        Here's the transcription to process:
''' + text

        # Make multiple attempts with different temperatures
        temperatures = [0.2, 0.3, 0.4, 0.5, 0.6]  # More temperature variations
        responses = []
        
        for temp in temperatures:
            try:
                response = self._send_to_ollama(prompt, temperature=temp)
                parsed_response = self._validate_and_parse_response(response)
                # Add original transcription to each response
                parsed_response['original_transcription'] = text
                # Evaluate the conversion quality
                evaluation = self._evaluate_conversion(text, parsed_response)
                parsed_response['conversion_quality'] = evaluation
                responses.append(parsed_response)
                logging.info(f"Successfully generated response with temperature {temp}")
            except Exception as e:
                logging.warning(f"Failed attempt with temperature {temp}: {str(e)}")
                continue
        
        if not responses:
            # If all attempts fail, return a basic response with the original text
            return {
                "title": "Untitled Note",
                "content": text,
                "tags": [],
                "tasks": [],
                "original_transcription": text,
                "conversion_quality": {
                    "completeness": 0,
                    "structure": 0,
                    "clarity": 0,
                    "task_handling": 0,
                    "overall_quality": 0,
                    "explanation": "Conversion failed"
                }
            }
        
        # Compare and select the best response
        best_response = self._compare_responses(responses)
        # Ensure original transcription is in the best response
        best_response['original_transcription'] = text
        return best_response
