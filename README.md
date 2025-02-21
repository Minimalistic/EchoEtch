# EchoEtch

An automated system that converts audio notes into formatted Markdown notes with allowed tags using local AI processing.

## Features

- Monitors a folder for new audio files
- Transcribes audio using Whisper (local)
- Processes transcriptions with Ollama (local)
- Auto-generates formatted Markdown notes with tags
- Links original audio files in Obsidian vault
- Tag management system with allowed tags support
- Completely local processing - no cloud services required

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally
- FFmpeg installed for audio processing
- Obsidian vault set up

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

### Allowed Tags File

The `allowed_tags.md` file defines the list of tags that can be used in your notes. This helps maintain consistency and organization in your note-taking system.

#### Location
By default, the allowed tags file should be placed in your Obsidian vault's root directory. The location is specified by the `ALLOWED_TAGS_FILE` setting in the `.env` file.

#### File Format
Create a Markdown file with a list of allowed tags, one per line. Each tag should:
- Start with a `#`
- Use lowercase letters
- Use hyphens for multi-word tags
- Optionally use hierarchical tags with `/`

Example `allowed_tags.md`:
```markdown
# Allowed Tags

#meeting
#idea
#project
#journal
#family
#biking
#bills
#books/fiction
#books/non-fiction
#health
#finance
#quote
#career
#gaming
#rant
```

#### Purpose
- Restricts tags to a predefined list to prevent hallucinations
- Ensures consistent tag usage across notes
- Helps with organization and searchability
- Can be easily modified as your note-taking needs evolve
