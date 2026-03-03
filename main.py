import os
import sys
from pathlib import Path
from dotenv import load_dotenv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    # 1. Define paths relative to this script's location
    root_dir = Path(__file__).parent.absolute()
    ui_path = root_dir / "app" / "ui" / "main_ui.py"

    # --- CRITICAL STEP: Load API keys from .env file ---
    load_dotenv(root_dir / ".env")

    if not ui_path.exists():
        print(f"ERROR: UI file not found at {ui_path}")
        sys.exit(1)

    # 2. Run Streamlit with forced PYTHONPATH to current directory
    # This ensures absolute imports work correctly
    os.environ["PYTHONPATH"] = str(root_dir)

    command = (
        f'streamlit run "{ui_path}" '
        f'--client.showErrorDetails=true '
        f'--server.fileWatcherType none'
    )

    print(f"Launching Lumina from: {root_dir}")
    os.system(command)