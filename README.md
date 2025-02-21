# EchoEtch

An automated system that converts audio notes into formatted Markdown notes with allowed tags using local AI processing.

## Features

- Monitors a selected folder for new audio files
- Transcribes audio locally using Whisper
- Processes transcriptions locally with Ollama
- Auto-generates formatted Markdown notes with allowed tags
- Links original audio files in Obsidian vault
- Completely local processing - no cloud services required

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally
- FFmpeg installed for audio processing
- Obsidian vault set up

## File Processing and Storage

### Notes Folder Structure

The `Daily Notes` folder is used for processed files with the following conventions:

- A sub-folder `audio` is created to store processed audio files
- Markdown files are generated for each processed audio file
- A markdown file is saved in the `Daily Notes` folder for each processed audio file and follows the format: `yyyy-MM-dd-Succinct-Generated-File-Name.md`

## Setup

1. Clone this repository
2. (Optional) Create and activate a virtual environment:
   ```bash
   # Using venv (recommended for most users)
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   
   # Alternative: Use conda or your preferred environment manager
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and configure your paths
5. With Ollama.ai installed locally, open a new terminal and run `ollama pull mistral` (or whatever model you want to use)
6. Ensure the Ollama model you want to use is set in the `.env` file (By default the example .env is set to `mistral`)
7. Ensure Ollama is running locally by running `ollama serve`
8. Run the watcher:
   ```bash
   python main.py
   ```
## Configuration

Edit the `.env` file to configure:
- Audio input folder path
- Obsidian vault path
- Ollama model name and settings
- Note template settings
- Allowed tags file location
