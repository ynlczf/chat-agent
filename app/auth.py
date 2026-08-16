"""
Single-user session auth with in-memory sessions + bcrypt password hash.
Anti-brute-force: per-IP sliding-window failure tracking with exponential
lockout.  No registration — account is configured via AUTH_USERNAME /
AUTH_PASSWORD env vars (set once in .env).

If AUTH_USERNAME or AUTH_PASSWORD is empty, auth is disabled: every
request.state.user stays None, and routes fall back to legacy defaults.
"""
from __future__ import annotations

import logging
import secrets
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import bcrypt as _bcrypt
from fastapi import APIRouter, HTTPException, Request, Depends

from .config import (
    AUTH_USERNAME, AUTH_PASSWORD, AUTH_ORG_ID,
    AUTH_MAX_LOGIN_ATTEMPTS, AUTH_LOCKOUT_SECONDS, AUTH_TOKEN_TTL_HOURS,
)
from .models import LoginRequest, AuthResponse, UserResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# bcrypt work factor (4..31). 12 is the OWASP-recommended default;
# single-user local service — anything ≥10 is fine.
_BCRYPT_ROUNDS = 12

# ---------------------------------------------------------------------------
# Runtime state (process-wide, in-memory)
# ---------------------------------------------------------------------------

# Hashed password computed once at module load; plaintext never kept.
_hashed_password: Optional[str] = None

# token -> (user_response, expires_at_epoch)
_sessions: Dict[str, tuple] = {}
_sessions_lock = threading.Lock()

# Anti-brute-force: ip -> {"fails": [timestamps...], "lock_until": epoch}
_attempts: Dict[str, Dict] = {}
_attempts_lock = threading.Lock()

TOKEN_TTL_SECONDS = AUTH_TOKEN_TTL_HOURS * 3600


def _auth_enabled() -> bool:
    return bool(AUTH_USERNAME and AUTH_PASSWORD)


def _init_password_hash() -> None:
    """Hash AUTH_PASSWORD once at import time (or on first use).

    bcrypt 5.x returns bytes; store as utf-8 string for logging/debug.
    """
    global _hashed_password
    if _hashed_password is None and AUTH_PASSWORD:
        hashed = _bcrypt.hashpw(
            AUTH_PASSWORD.encode("utf-8"),
            _bcrypt.gensalt(rounds=_BCRYPT_ROUNDS),
        )
        _hashed_password = hashed.decode("utf-8")
        logger.info("Auth: single-user password hash loaded (username=%s)", AUTH_USERNAME)


_init_password_hash()


def _verify_password(plain: str, hashed: str) -> bool:
    """Constant-time bcrypt verify. Safe with plaintext longer than 72 bytes
    (bcrypt 5.x auto-truncates)."""
    try:
        return _bcrypt.checkpw(
            plain.encode("utf-8"),
            hashed.encode("utf-8"),
        )
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Anti-brute-force
# ---------------------------------------------------------------------------

def _client_ip(request: Request) -> str:
    """Get best-effort client IP (supports X-Forwarded-For behind a proxy)."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_lockout(ip: str) -> Optional[float]:
    """Return remaining lockout seconds if IP is locked, else None."""
    with _attempts_lock:
        rec = _attempts.get(ip)
        if rec and rec.get("lock_until", 0) > time.time():
            remaining = rec["lock_until"] - time.time()
            return max(0.0, remaining)
        return None


def _record_failure(ip: str) -> float:
    """Increment failure counter; lock out IP if threshold crossed.
    Returns lockout remaining seconds (0 if not locked yet)."""
    now = time.time()
    with _attempts_lock:
        rec = _attempts.setdefault(ip, {"fails": []})
        # Keep only failures within the last 5 minutes (sliding window)
        rec["fails"] = [t for t in rec["fails"] if now - t < 300]
        rec["fails"].append(now)

        # Exponential backoff: lockout = base * (2 ** (consecutive_blocks))
        if len(rec["fails"]) >= AUTH_MAX_LOGIN_ATTEMPTS:
            prev_lock = rec.get("prev_lock_base", AUTH_LOCKOUT_SECONDS)
            lock_seconds = min(prev_lock * 2, 3600)  # cap at 1h
            rec["lock_until"] = now + lock_seconds
            rec["prev_lock_base"] = lock_seconds
            logger.warning(
                "Auth: IP %s locked out for %.0fs after %d failures",
                ip, lock_seconds, len(rec["fails"]),
            )
            # Reset counter but keep lock
            rec["fails"] = []
            return lock_seconds
        return 0.0


def _clear_failures(ip: str) -> None:
    with _attempts_lock:
        _attempts.pop(ip, None)


def _cleanup_expired_sessions() -> None:
    now = time.time()
    with _sessions_lock:
        expired = [t for t, (_, exp) in _sessions.items() if exp < now]
        for t in expired:
            del _sessions[t]


# ---------------------------------------------------------------------------
# User / session helpers
# ---------------------------------------------------------------------------

def _build_user() -> UserResponse:
    """Return the single configured user."""
    return UserResponse(
        user_id="u_local",
        username=AUTH_USERNAME,
        org_id=AUTH_ORG_ID,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _create_session() -> str:
    _cleanup_expired_sessions()
    token = secrets.token_urlsafe(32)
    expires = time.time() + TOKEN_TTL_SECONDS
    with _sessions_lock:
        _sessions[token] = (_build_user(), expires)
    return token


def _lookup_session(token: str) -> Optional[UserResponse]:
    _cleanup_expired_sessions()
    with _sessions_lock:
        entry = _sessions.get(token)
        if entry is None:
            return None
        user, expires = entry
        if expires < time.time():
            del _sessions[token]
            return None
        return user


def _delete_session(token: str) -> None:
    with _sessions_lock:
        _sessions.pop(token, None)


# ---------------------------------------------------------------------------
# Token extraction / middleware entry point
# ---------------------------------------------------------------------------

def _extract_token(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return request.query_params.get("token") or None


def inject_user_to_request(request: Request) -> None:
    """Populate request.state.user from Bearer token.
    Always succeeds — if auth is disabled, missing, or invalid,
    request.state.user stays None and routes use legacy defaults."""
    if not _auth_enabled():
        request.state.user = None
        return
    token = _extract_token(request)
    if not token:
        request.state.user = None
        return
    request.state.user = _lookup_session(token)


def _resolve_current_user(request: Request) -> UserResponse:
    """FastAPI dependency: 401 if not authenticated."""
    if not _auth_enabled():
        # Auth disabled — treat as if always logged in as default user
        return _build_user()
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    return user


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------

@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest, request: Request):
    if not _auth_enabled():
        raise HTTPException(
            status_code=503,
            detail="认证未配置。请在 .env 中设置 AUTH_USERNAME 和 AUTH_PASSWORD",
        )

    ip = _client_ip(request)

    # 1. Lockout check
    remaining = _check_lockout(ip)
    if remaining is not None:
        raise HTTPException(
            status_code=429,
            detail=f"尝试次数过多，请在 {int(remaining)} 秒后重试",
        )

    username = req.username.strip()
    password = req.password
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    # 2. Constant-timing-ish: always run a verify call even if user wrong
    user_ok = secrets.compare_digest(username, AUTH_USERNAME)
    pw_ok = _hashed_password is not None and _verify_password(password, _hashed_password)

    if not (user_ok and pw_ok):
        lock_secs = _record_failure(ip)
        if lock_secs > 0:
            raise HTTPException(
                status_code=429,
                detail=f"用户名或密码错误；尝试次数过多，已被锁定 {int(lock_secs)} 秒",
            )
        # Leak back a generic 401 — no hint which field was wrong
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 3. Success
    _clear_failures(ip)
    token = _create_session()
    logger.info("User logged in: username=%s, ip=%s", AUTH_USERNAME, ip)
    return AuthResponse(token=token, user=_build_user())


@router.post("/logout")
def logout(request: Request):
    token = _extract_token(request)
    if token:
        _delete_session(token)
    return {"status": "ok"}


@router.get("/me", response_model=UserResponse)
def me(current_user: UserResponse = Depends(_resolve_current_user)):
    return current_user
