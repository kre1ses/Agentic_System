"""
Standalone Kaggle dataset downloader.

Usage:
    python authenticate.py                              # default competition
    python authenticate.py mws-ai-agents-2026          # explicit competition name
    python authenticate.py mws-ai-agents-2026 ./data_2 # + custom output dir

Credentials:
    The script reads credentials from ~/.kaggle/kaggle.json or
    KAGGLE_USERNAME + KAGGLE_KEY environment variables.
    If neither is present, it prompts for the API-token JSON
    (downloaded from kaggle.com → Settings → API → Create New Token).

This script delegates all logic to main.kaggle_download().
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import kaggle_download

if __name__ == "__main__":
    competition = sys.argv[1] if len(sys.argv) > 1 else "mws-ai-agents-2026"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./data_2"
    kaggle_download(competition, output_dir)
