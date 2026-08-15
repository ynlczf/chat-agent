import contextvars
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

from deepagents.backends import StateBackend
from deepagents.backends.protocol import SandboxBackendProtocol, ExecuteResponse
from deepagents_opensandbox import OpensandboxBackend
from opensandbox.sync.sandbox import SandboxSync

from .config import SANDBOX_IDLE_TIMEOUT, SANDBOX_CLEANUP_INTERVAL, PPT_SKILL_DIR, PPT_CORE_DEPS

logger = logging.getLogger(__name__)


class _SandboxEntry:
    def __init__(self, sandbox: SandboxSync, backend: OpensandboxBackend):
        self.sandbox = sandbox
        self.backend = backend
        self.last_used_at = time.time()
        self.lock = threading.Lock()
        self.provisioned = False
        self.provisioning_error: Optional[str] = None

    def touch(self):
        self.last_used_at = time.time()

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used_at

    def close(self):
        try:
            self.sandbox.kill()
        except Exception:
            pass
        try:
            self.sandbox.close()
        except Exception:
            pass


def _provision_sandbox_with_ppt_skill(backend: OpensandboxBackend) -> Tuple[bool, str]:
    if not PPT_SKILL_DIR.exists():
        return False, f"local skill dir not found: {PPT_SKILL_DIR}"

    upload_pairs: List[Tuple[str, bytes]] = []
    skip_dirs = {"__pycache__"}
    for root, dirs, files in os.walk(PPT_SKILL_DIR):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            if fname.endswith(".pyc") or fname.startswith(".env"):
                continue
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, PPT_SKILL_DIR).replace("\\", "/")
            sandbox_path = f"/skills/ynlczf-pptx-generator/{rel}"
            try:
                with open(local_path, "rb") as f:
                    upload_pairs.append((sandbox_path, f.read()))
            except Exception as e:
                logger.debug(f"Provisioning: skip unreadable {local_path}: {e}")

    BATCH = 50
    failed = 0
    for i in range(0, len(upload_pairs), BATCH):
        chunk = upload_pairs[i:i + BATCH]
        try:
            responses = backend.upload_files(chunk)
            for r in responses:
                if r.error:
                    failed += 1
        except Exception as e:
            logger.warning(f"Provisioning: upload batch failed: {e}")
            failed += len(chunk)

    try:
        backend.execute(
            "python3 -m pip --version >/dev/null 2>&1 || "
            "("
            "apt-get update -qq && apt-get install -y -qq python3-pip >/dev/null 2>&1"
            ") || "
            "("
            "curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py 2>/dev/null "
            "|| wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py 2>/dev/null"
            ") && python3 /tmp/get-pip.py --quiet --break-system-packages 2>/dev/null"
            " || python3 -m ensurepip --upgrade --break-system-packages 2>/dev/null"
            " || true",
            timeout=120,
        )
        install_res = backend.execute(
            "python3 -m pip install --quiet --disable-pip-version-check --break-system-packages "
            + " ".join(PPT_CORE_DEPS) + " 2>&1",
            timeout=300,
        )
        deps_ok = install_res.exit_code == 0
    except Exception as e:
        logger.warning(f"Provisioning: dep install error: {e}")
        deps_ok = False
        install_res = None

    deps_tail = ""
    if install_res is not None and not deps_ok:
        deps_tail = " | pip: " + (install_res.output or "")[-300:].replace("\n", " ")

    msg = (
        f"uploaded {len(upload_pairs)} files ({failed} failed); "
        f"deps install {'ok' if deps_ok else 'failed'}{deps_tail}"
    )
    return deps_ok, msg


class SandboxManager:
    def __init__(self, image: str, idle_timeout: int = SANDBOX_IDLE_TIMEOUT,
                 cleanup_interval: int = SANDBOX_CLEANUP_INTERVAL):
        self._image = image
        self._idle_timeout = idle_timeout
        self._cleanup_interval = cleanup_interval
        self._entries: Dict[str, _SandboxEntry] = {}
        self._global_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._available = False

    def start(self):
        probe = None
        try:
            logger.info("SandboxManager: probing sandbox availability...")
            probe = SandboxSync.create(self._image)
            probe.kill()
            probe.close()
            self._available = True
            logger.info("SandboxManager: sandbox available, starting cleanup thread.")
        except Exception as e:
            logger.warning(f"SandboxManager: sandbox unavailable — {e}")
            self._available = False
            return

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, name="sandbox-cleanup", daemon=True,
        )
        self._cleanup_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)

        with self._global_lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for e in entries:
            with e.lock:
                e.close()
        logger.info(f"SandboxManager: stopped, cleaned up {len(entries)} sandbox(es).")

    @property
    def available(self) -> bool:
        return self._available

    def _cleanup_loop(self):
        logger.info(
            f"SandboxManager: cleanup thread started "
            f"(idle_timeout={self._idle_timeout}s, check_interval={self._cleanup_interval}s)."
        )
        while not self._stop_event.is_set():
            try:
                self._reap_expired()
            except Exception as e:
                logger.error(f"SandboxManager: cleanup error: {e}", exc_info=True)
            self._stop_event.wait(self._cleanup_interval)
        logger.info("SandboxManager: cleanup thread stopped.")

    def _reap_expired(self):
        to_remove: List[Tuple[str, _SandboxEntry]] = []
        with self._global_lock:
            items = list(self._entries.items())

        if not items:
            logger.debug("SandboxManager: cleanup scan — no active sandboxes.")
            return

        for org_id, entry in items:
            with entry.lock:
                idle = entry.idle_seconds
            remaining = self._idle_timeout - idle
            if remaining <= 0:
                logger.info(
                    f"SandboxManager: [SCAN] org={org_id} idle={idle:.0f}s "
                    f"(>= threshold {self._idle_timeout}s) → WILL RECLAIM"
                )
                to_remove.append((org_id, entry))
            else:
                logger.debug(
                    f"SandboxManager: [SCAN] org={org_id} idle={idle:.0f}s "
                    f"(recycle in {remaining:.0f}s)"
                )

        if not to_remove:
            logger.debug(
                f"SandboxManager: cleanup scan complete — "
                f"{len(items)} sandbox(es) active, 0 reclaimed."
            )
            return

        reclaimed = 0
        for org_id, entry in to_remove:
            with self._global_lock:
                cur = self._entries.get(org_id)
                if cur is entry:
                    del self._entries[org_id]
                else:
                    logger.warning(
                        f"SandboxManager: org={org_id} entry changed before "
                        f"reclamation, skipping."
                    )
                    continue
            with entry.lock:
                idle_val = entry.idle_seconds
                entry.close()
            reclaimed += 1
            logger.info(
                f"SandboxManager: [RECLAIMED] org={org_id} sandbox destroyed "
                f"(was idle {idle_val:.0f}s)."
            )

        still_active = len(items) - reclaimed
        logger.info(
            f"SandboxManager: cleanup scan complete — "
            f"scanned={len(items)}, reclaimed={reclaimed}, still_active={still_active}."
        )

    def get_backend(self, org_id: str) -> Optional[OpensandboxBackend]:
        if not self._available:
            return None

        with self._global_lock:
            entry = self._entries.get(org_id)
            if entry is None:
                logger.info(f"SandboxManager: creating new sandbox for org={org_id}")
                sandbox = SandboxSync.create(self._image)
                backend = OpensandboxBackend(sandbox=sandbox)
                entry = _SandboxEntry(sandbox=sandbox, backend=backend)
                self._entries[org_id] = entry
                need_provision = True
            else:
                logger.debug(f"SandboxManager: reusing existing sandbox for org={org_id}")
                need_provision = False

        with entry.lock:
            entry.touch()
            backend = entry.backend

        if need_provision:
            t = threading.Thread(
                target=self._provision_entry,
                args=(org_id, entry),
                name=f"ppt-provision-{org_id}",
                daemon=True,
            )
            t.start()

        return backend

    @staticmethod
    def _provision_entry(org_id: str, entry: "_SandboxEntry"):
        try:
            ok, msg = _provision_sandbox_with_ppt_skill(entry.backend)
            with entry.lock:
                entry.provisioned = ok
                entry.provisioning_error = None if ok else msg
            logger.info(f"SandboxManager: PPT provisioning for org={org_id} → {msg}")
        except Exception as e:
            with entry.lock:
                entry.provisioned = False
                entry.provisioning_error = str(e)
            logger.warning(f"SandboxManager: PPT provisioning failed for org={org_id}: {e}")

    def get_sandbox_count(self) -> int:
        with self._global_lock:
            return len(self._entries)


_org_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "org_id", default=None
)


def _set_current_org(org_id: Optional[str]):
    _org_id_var.set(org_id)


def _get_current_org() -> Optional[str]:
    return _org_id_var.get()


_sandbox_manager: Optional[SandboxManager] = None


def init_sandbox_manager(image: str) -> SandboxManager:
    global _sandbox_manager
    _sandbox_manager = SandboxManager(image)
    _sandbox_manager.start()
    return _sandbox_manager


def get_sandbox_manager() -> Optional[SandboxManager]:
    return _sandbox_manager


def _sandbox_available() -> bool:
    return _sandbox_manager is not None and _sandbox_manager.available


def _get_org_backend(org_id: Optional[str] = None) -> Optional[OpensandboxBackend]:
    if _sandbox_manager is None or not _sandbox_manager.available:
        return None
    target = org_id if org_id is not None else _get_current_org()
    if target is None:
        return None
    return _sandbox_manager.get_backend(target)


class _OrgScopedSandboxBackendProxy:
    def __init__(self, manager: SandboxManager):
        self._manager = manager
        self._fallback = StateBackend()

    def _resolve(self):
        if not self._manager.available:
            logger.warning(
                f"SandboxProxy: sandbox unavailable, using fallback StateBackend "
                f"(thread={threading.get_ident()})."
            )
            return self._fallback
        org_id = _get_current_org()
        if org_id is None:
            logger.warning(
                f"SandboxProxy: thread-local org_id is None (thread={threading.get_ident()}). "
                f"Falling back to StateBackend — agent tool calls will NOT hit a sandbox."
            )
            return self._fallback
        backend = self._manager.get_backend(org_id)
        if backend is None:
            logger.warning(
                f"SandboxProxy: get_backend returned None for org={org_id} "
                f"(thread={threading.get_ident()})."
            )
            return self._fallback
        return backend

    @property
    def id(self) -> str:
        backend = self._resolve()
        return getattr(backend, "id", "proxy")

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        backend = self._resolve()
        if hasattr(backend, "execute"):
            return backend.execute(command, timeout=timeout)
        raise NotImplementedError(
            "Resolved backend does not support command execution."
        )

    def __getattr__(self, name):
        underlying = self._resolve()
        return getattr(underlying, name)


SandboxBackendProtocol.register(_OrgScopedSandboxBackendProxy)
