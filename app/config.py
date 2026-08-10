import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("DB_URI", "postgresql://postgres:root@127.0.0.1:5432/testdb")

SANDBOX_IMAGE = os.getenv(
    "SANDBOX_IMAGE",
    "opensandbox/code-interpreter:v1.1.0",
)
SANDBOX_IDLE_TIMEOUT = 10 * 60
SANDBOX_CLEANUP_INTERVAL = 60

PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_SKILLS_DIR = PROJECT_ROOT / "skills"
PPT_SKILL_DIR = LOCAL_SKILLS_DIR / "ynlczf-pptx-generator"
PPT_CORE_DEPS = ["python-pptx"]

MODEL_NAME = "deepseek:deepseek-v4-flash"
