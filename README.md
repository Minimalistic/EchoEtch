# TalkToNote

An automated system that converts audio notes into well-formatted Markdown notes using local AI processing.

## Features

- Monitors a folder for new audio files
- Transcribes audio using Whisper (local)
- Processes transcriptions with Ollama (local)
- Auto-generates formatted Markdown notes with tags
- Links original audio files in Obsidian vault
- Completely local processing - no cloud services required

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally
- FFmpeg installed for audio processing
- Obsidian vault set up

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and configure your paths
4. Run the watcher:
   ```bash
   python main.py
   ```

## Configuration

Edit the `.env` file to configure:
- Audio input folder path
- Obsidian vault path
- Ollama model name (default: mistral)
- Note template settings
