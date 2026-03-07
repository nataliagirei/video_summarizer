# Lumina AI Assistant

Lumina AI Assistant is a Python-based application that extracts, transcribes, and analyzes content from video and audio files. It leverages AI models for speech recognition (Whisper), text analysis, and optional visual analysis of video frames. The app provides an intuitive web interface powered by Streamlit, enabling users to interact with AI, generate summaries, and export reports.

## Features

- **Video to Transcript**: Extracts audio from video and generates accurate transcriptions.
- **Multi-language UI**: Supports English, Russian, and Polish.
- **AI Chat**: Ask questions about the content of processed videos and audios.
- **Draft Editor**: Save, edit, and export transcripts or summaries as PDF.
- **Visual Analysis**: Link transcript segments to video frames for context.
- **Retrieval-Augmented QA**: Query video content using advanced retrieval techniques.
- **Flexible Input**: Upload files, use live microphone recording, or provide YouTube URLs.
- **Export Options**: Download transcripts, summaries, and ZIP packages containing processed data.

## Project Structure

**video_summarizer/**  
├── `.venv/` – Python virtual environment  
├── `app/` – Presentation layer (User Interface)  
│   └── `ui/`  
│       ├── `__init__.py`  
│       ├── `chat.py` – AI chat handling, allows querying of processed content  
│       ├── `editor.py` – Draft editor for transcripts and summaries  
│       ├── `main_ui.py` – Main Streamlit interface  
│       └── `ui_lang.py` – Multi-language UI dictionary  
├── `assets/` – Static files (icons, images, fonts)  
├── `data/` – Storage for uploaded files, transcripts, and processed data  
│   ├── `raw/`  
│   ├── `audio/`  
│   ├── `transcripts/`  
│   ├── `processed/`  
│   ├── `reports/`  
│   └── `frames/`  
├── `infrastructure/` – Technical layer, handles external interactions  
│   ├── `utils/`  
│   │   ├── `__init__.py`  
│   │   └── `utils.py` – Generic helper functions  
│   ├── `audio_extractor.py` – Extracts audio streams from video files  
│   ├── `frame_extractor.py` – Extracts frames from videos for visual analysis  
│   ├── `microphone_recorder.py` – Captures live audio via microphone  
│   ├── `vector_store.py` – Manages FAISS vector database for semantic search  
│   ├── `whisper_client.py` – Wrapper for Whisper ASR model  
│   └── `youtube_client.py` – Downloads YouTube videos  
├── `services/` – Business logic (core application)  
│   ├── `processing/` – Text processing and export  
│   │   ├── `export.py` – General export utilities  
│   │   ├── `export_pdf.py` – PDF report generation  
│   │   ├── `text_chunk.py` – Splits large transcripts into chunks  
│   │   ├── `text_clean.py` – Cleans and normalizes raw text  
│   │   ├── `transcript_processor.py` – Core transcription logic  
│   │   └── `translation.py` – Translation for multi-language support  
│   ├── `rag/` – Retrieval-Augmented Generation (RAG)  
│   │   └── `rag_service.py` – AI question-answering based on content  
│   └── `vision/` – Visual analysis modules  
│       └── `visual_linker.py` – Links video frames with transcript segments  
├── `.env` – Environment variables (API keys, configs)  
├── `.gitignore` – Git exclusions (.venv, __pycache__, etc.)  
├── `config.yaml` – Application configuration  
├── `main.py` – Application entry point  
├── `pipeline.py` – Orchestrates end-to-end process (E2E)  
├── `README.md` – Project documentation  
└── `requirements.txt` – Python dependencies  

## Technical Details / Architecture

### Pipeline Overview
The `Pipeline` class orchestrates the end-to-end processing of video content:

1. **Input Handling**:
   - **File Upload**: Accepts video/audio files from local storage.
   - **Microphone Recording**: Captures live audio.
   - **YouTube URL**: Downloads video via `youtube_client.py`.

2. **Audio Processing**:
   - Extracts audio from video files using `audio_extractor.py`.
   - Supports preprocessing and cleaning through `text_clean.py`.
   - Optional live recording via `microphone_recorder.py`.

3. **Speech Recognition (ASR)**:
   - Utilizes the Whisper model (`whisper_client.py`) for accurate transcription.
   - Supports multiple languages and can switch model size for performance vs. accuracy.

4. **Text Processing**:
   - **Chunking**: Splits transcript into manageable chunks (`text_chunk.py`).
   - **Translation**: Optional translation using `translation.py`.
   - **Transcript Processor**: Core logic for handling, cleaning, and storing transcripts.

5. **Retrieval-Augmented Generation (RAG)**:
   - Implements semantic search and QA on transcript content (`rag/rag_service.py`).
   - Uses FAISS vector database (`vector_store.py`) to efficiently index and query embeddings.

6. **Visual Analysis**:
   - Extracts frames from video (`frame_extractor.py`) for visual context.
   - Links textual content to video frames (`vision/visual_linker.py`), enabling synchronized transcript review.

7. **Business Logic / Services**:
   - Processing, export, and PDF report generation handled in `services/processing/`.
   - RAG and Vision modules are separate for modular extension.

8. **User Interface (Streamlit)**:
   - Multi-language sidebar with input selection, model quality, and draft management (`app/ui/main_ui.py`).
   - AI chat for contextual questions (`chat.py`).
   - Draft editor for managing summaries and transcripts (`editor.py`).
   - Transcript display with optional original text toggle.


## Installation

1. Clone the repository:


git clone https://github.com/nataliagirei/video_summarizer.git
cd video_summarizer

2. Create a virtual environment and activate it:

# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate

3. Install dependencies:

pip install -r requirements.txt

4. Add environment variables to `.env`:

API_KEY=your_api_key_here

5. Run the application:

run main.py

## Usage

- Select your interface language: English, Russian, or Polish.
- Choose model quality: Fast (lightweight) or High (more accurate).
- Upload a video file, record live audio, or enter a YouTube URL.
- Optionally enable visual analysis to link transcript text with video frames.
- Use the AI chat to ask questions about the content.
- Generate summaries or key insights using the quick analysis buttons.
- Save or edit drafts in the Draft Editor and export as PDF if needed.
- Download ZIP packages containing transcripts, summaries, and associated files.