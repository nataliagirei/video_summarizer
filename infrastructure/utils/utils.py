import json
import os
import tempfile
from datetime import datetime
from pathlib import Path


# --- TIME UTILS ---
def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def current_datetime_str(fmt="%Y-%m-%d %H:%M:%S") -> str:
    """Return current datetime as string."""
    return datetime.now().strftime(fmt)


# --- JSON UTILS ---
def load_json(file_path: Path) -> dict:
    """
    Safely load JSON from file.
    Returns empty dict if file does not exist or contains invalid JSON.
    """
    if not file_path.exists():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON decode error in {file_path}: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error reading {file_path}: {e}")
        return {}


def save_json(file_path: Path, data: dict):
    """
    Save dictionary as JSON atomically.
    Uses temporary file and os.replace to avoid corruption if interrupted.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(file_path.parent)) as tmp:
            json.dump(data, tmp, ensure_ascii=False, indent=2)
            temp_name = tmp.name
        os.replace(temp_name, file_path)
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")


# --- TEXT / HTML UTILS ---
def prepare_content_for_editor(text: str) -> str:
    """Prepare text for Quill/HTML editor (replace line breaks)."""
    return text.replace("\n", "<br>")


def add_snippet_to_content(content: str, timestamp: float, snippet_text: str) -> str:
    """Append snippet to HTML content with timestamp formatting."""
    snippet = f"<p><b>[{format_timestamp(timestamp)}]</b> {snippet_text}</p>"
    return content + snippet


# --- FILE UTILS ---
def delete_files_by_prefix(folder: Path, prefix: str, silent: bool = True):
    """
    Delete all files in folder starting with prefix.
    If silent=False, prints errors.
    """
    if not folder.exists():
        return
    for file in folder.glob(f"{prefix}*"):
        try:
            file.unlink()
        except Exception as e:
            if not silent:
                print(f"Error deleting file {file}: {e}")


# --- REGISTRY / DRAFT UTILS ---
def delete_source_data(
        source_id: str,
        data_dirs: dict,
        drafts_file: Path = None,
        registry_file: Path = None,
        silent: bool = True
):
    """
    Delete all source-related data from registry, drafts, and data directories.
    """
    # Delete from registry
    if registry_file and registry_file.exists():
        registry = load_json(registry_file)
        if source_id in registry:
            registry.pop(source_id)
            save_json(registry_file, registry)

    # Delete from drafts
    if drafts_file and drafts_file.exists():
        drafts = load_json(drafts_file)
        if source_id in drafts:
            drafts.pop(source_id)
            save_json(drafts_file, drafts)

    # Delete all files in data directories
    for folder in data_dirs.values():
        delete_files_by_prefix(folder, source_id, silent=silent)
