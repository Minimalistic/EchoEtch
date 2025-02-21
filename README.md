# EchoEtcher

An automated system that converts audio notes into formatted Markdown notes with allowed tags using local AI processing.

## Project Status ⚠️

**Experimental / Work in Progress**

This project is a rapid prototype developed through collaborative AI assistance and personal iteration. As such, it comes with some important caveats:

- 🧪 Experimental: The codebase is in active development
- 🛠 Unstable: Expect potential breaking changes
- 🐛 Limited Testing: Minimal comprehensive testing has been performed
- 🚧 Use at Your Own Risk: Not recommended for production environments without significant review and modification

Contributions, feedback, and improvements are welcome! If you encounter issues or have suggestions, please open an issue on the repository.

## Disclaimer

By using EchoEtcher, you acknowledge that you understand and accept the risks associated with using an experimental project. You agree to hold harmless the developers and contributors of EchoEtcher for any damages or losses resulting from its use.

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

### Tag Processing

When processing tags from the transcribed audio, EchoEtcher uses a unique approach to tag management:

- The system allows you to specify a set of `allowed_tags`
- When writing the final tags to the Markdown file, these tags are **prepended** with '#echo-etcher/' 
- Prepending makes it easier to filter and search for specific tags that were created by EchoEtcher in your notes
- This approach ensures consistency and improves note organization and discoverability

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
