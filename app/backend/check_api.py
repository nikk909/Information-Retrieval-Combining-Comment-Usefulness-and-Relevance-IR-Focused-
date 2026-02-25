import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent.parent / ".env"

load_dotenv(_env_path)

api_key = os.environ.get("YOUTUBE_API_KEY")

print(api_key)