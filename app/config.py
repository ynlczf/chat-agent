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

# ---------------------------------------------------------------------------
# Auth — single-user mode. Set in .env. If AUTH_USERNAME or AUTH_PASSWORD
# is missing, auth is effectively disabled (all endpoints fall back to
# legacy local-user / default-org defaults).
# ---------------------------------------------------------------------------
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "")
AUTH_ORG_ID = os.getenv("AUTH_ORG_ID", "default-org")

# Anti-brute-force settings
AUTH_MAX_LOGIN_ATTEMPTS = int(os.getenv("AUTH_MAX_LOGIN_ATTEMPTS", "5"))
AUTH_LOCKOUT_SECONDS = int(os.getenv("AUTH_LOCKOUT_SECONDS", "300"))   # 5 min per lockout
AUTH_TOKEN_TTL_HOURS = int(os.getenv("AUTH_TOKEN_TTL_HOURS", "168"))   # 7 days
