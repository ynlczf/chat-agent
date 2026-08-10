from contextlib import asynccontextmanager, ExitStack
import re
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import os
import asyncio
import contextvars
import json
import threading
import time
import uvicorn
from pathlib import Path
import logging

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StoreBackend, StateBackend, FilesystemBackend
from deepagents.backends.protocol import SandboxBackendProtocol, ExecuteResponse
from deepagents_opensandbox import OpensandboxBackend
from opensandbox.sync.sandbox import SandboxSync
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver

from context_type import MemoryContext
from namespace_router import assistant_namespace, user_namespace
from tools.file_handler import FileHandler, ParsedFile

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

DB_URI = os.getenv("DB_URI", "postgresql://postgres:root@127.0.0.1:5432/testdb")

SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "opensandbox/code-interpreter:v1.1.0")
SANDBOX_IDLE_TIMEOUT = 10 * 60  # 10 minutes in seconds
SANDBOX_CLEANUP_INTERVAL = 60   # Check every 60 seconds

# Local skill directory — served read-only to the agent via a FilesystemBackend
# route on /skills/. The ynlczf-pptx-generator scripts are also uploaded into
# each org's sandbox so `execute` can run them with SKILL_DIR=/skills/ynlczf-pptx-generator.
PROJECT_ROOT = Path(__file__).parent
LOCAL_SKILLS_DIR = PROJECT_ROOT / "skills"
PPT_SKILL_DIR = LOCAL_SKILLS_DIR / "ynlczf-pptx-generator"
# Lightweight PPT deps (ynlczf-pptx-generator only needs python-pptx).
# Provisioned into each new sandbox.
PPT_CORE_DEPS = ["python-pptx"]

# Global references for resources
_store = None
_checkpointer = None
_agent = None
_sandbox_manager: Optional["SandboxManager"] = None
_uploaded_files: Dict[str, ParsedFile] = {}
# Track which org's sandbox a file was uploaded to, plus cache the raw bytes
# so we can re-upload if the sandbox was recycled (idle > 10 min).
_file_org_map: Dict[str, str] = {}        # file_id  -> org_id
_file_content_cache: Dict[str, bytes] = {} # file_id  -> raw file bytes


class _SandboxEntry:
    """Internal entry for a single org's sandbox."""
    def __init__(self, sandbox: SandboxSync, backend: OpensandboxBackend):
        self.sandbox = sandbox
        self.backend = backend
        self.last_used_at = time.time()
        self.lock = threading.Lock()
        # PPT skill provisioning state (populated by a background thread)
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
    """Upload the ynlczf-pptx-generator skill tree into the sandbox and install core deps.

    Mirrors the local ``skills/ynlczf-pptx-generator/`` directory to
    ``/skills/ynlczf-pptx-generator/`` inside the sandbox so that
    ``SKILL_DIR=/skills/ynlczf-pptx-generator`` resolves consistently when the
    agent runs skill scripts via ``execute``. Then bootstraps pip (if missing)
    and installs the lightweight python-pptx dependency.

    This is best-effort and non-fatal: any failure is logged and reported so
    the sandbox remains usable (the agent can pip-install missing packages on
    demand).
    """
    if not PPT_SKILL_DIR.exists():
        return False, f"local skill dir not found: {PPT_SKILL_DIR}"

    # 1. Walk the local ynlczf-pptx-generator tree and collect (sandbox_path, content).
    upload_pairs: List[Tuple[str, bytes]] = []
    skip_dirs = {"__pycache__"}
    for root, dirs, files in os.walk(PPT_SKILL_DIR):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            if fname.endswith(".pyc") or fname.startswith(".env"):
                continue
            local_path = Path(root) / fname
            rel = local_path.relative_to(PPT_SKILL_DIR).as_posix()
            sandbox_path = f"/skills/ynlczf-pptx-generator/{rel}"
            try:
                upload_pairs.append((sandbox_path, local_path.read_bytes()))
            except Exception as e:
                logger.debug(f"Provisioning: skip unreadable {local_path}: {e}")

    # 2. Batch-upload to the sandbox.
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

    # 3. Bootstrap pip if needed, then install core deps.
    try:
        backend.execute(
            "python3 -m pip --version >/dev/null 2>&1 || python3 -m ensurepip --upgrade >/dev/null 2>&1 || true",
            timeout=60,
        )
        install_res = backend.execute(
            "python3 -m pip install --quiet --disable-pip-version-check "
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
        # Surface a short tail of the pip output for debugging.
        deps_tail = " | pip: " + (install_res.output or "")[-300:].replace("\n", " ")

    msg = (
        f"uploaded {len(upload_pairs)} files ({failed} failed); "
        f"deps install {'ok' if deps_ok else 'failed'}{deps_tail}"
    )
    return deps_ok, msg


class SandboxManager:
    """Manages per-organization sandbox containers with idle auto-recycling.

    Policy:
      * One sandbox per org_id at most.
      * If an org's sandbox is idle for >= SANDBOX_IDLE_TIMEOUT (10 min), it is
        automatically destroyed and must be re-created on next use.
      * A background cleanup thread periodically evicts expired entries.
    """

    def __init__(self, image: str, idle_timeout: int = SANDBOX_IDLE_TIMEOUT,
                 cleanup_interval: int = SANDBOX_CLEANUP_INTERVAL):
        self._image = image
        self._idle_timeout = idle_timeout
        self._cleanup_interval = cleanup_interval
        self._entries: Dict[str, _SandboxEntry] = {}
        self._global_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._available = False  # Whether sandbox feature is usable at all

    def start(self):
        """Probe availability and start the background cleanup thread."""
        # Quick probe — if we can't create even one sandbox, mark unavailable.
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
        """Stop cleanup thread and destroy all sandboxes."""
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
        """Evict sandboxes idle longer than the timeout, with detailed logging."""
        to_remove: List[Tuple[str, _SandboxEntry]] = []
        with self._global_lock:
            items = list(self._entries.items())

        if not items:
            logger.debug("SandboxManager: cleanup scan — no active sandboxes.")
            return

        # Log the status of every active sandbox
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
                # Double-check ownership under global lock
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
        """Get or create the OpensandboxBackend for the given org.

        Returns None if the sandbox feature is unavailable.
        Thread-safe: multiple concurrent calls for the same org will return
        the same shared sandbox entry (one container per org).
        """
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

        # Acquire entry lock (not global) while touching
        with entry.lock:
            entry.touch()
            backend = entry.backend

        # Provision the PPT skill (upload scripts + install deps) in the
        # background so the first chat isn't blocked. By the time the agent
        # finishes reading SKILL.md / references and starts running scripts,
        # provisioning is typically done. If a scripts runs before completion,
        # the agent sees the error and can retry / pip-install on demand.
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
        """Background provisioning of a freshly created sandbox for PPT skill."""
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


# ---- ContextVar org context + dynamic backend proxy --------------------------
# The deep agent is built once with a "default backend" for tool execution.
# Since we need per-org sandboxes but the agent cannot swap backends per-call,
# we install a proxy backend that reads the current org from a ContextVar
# and delegates to the org-specific OpensandboxBackend from SandboxManager.
# Every sync helper (_sync_chat, _sync_upload_file, ...) is responsible for
# setting the org_id before doing work.
#
# IMPORTANT: We use contextvars.ContextVar (NOT threading.local()) because
# LangGraph's SyncPregelLoop dispatches tool execution via BackgroundExecutor,
# which submits tasks to a ThreadPoolExecutor using copy_context() + ctx.run().
# copy_context() propagates ContextVar values to the worker threads, but
# threading.local() is NOT propagated — so a thread-local org_id would be
# invisible during tool execution, causing the proxy to fall back to the
# (execution-incapable) StateBackend.
_org_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "org_id", default=None
)


def _set_current_org(org_id: Optional[str]):
    _org_id_var.set(org_id)


def _get_current_org() -> Optional[str]:
    return _org_id_var.get()


def _sandbox_available() -> bool:
    return _sandbox_manager is not None and _sandbox_manager.available


def _get_org_backend(org_id: Optional[str] = None) -> Optional[OpensandboxBackend]:
    """Resolve backend for an org. Falls back to thread-local org_id if not given.

    Returns None when sandbox feature is unavailable.
    """
    if _sandbox_manager is None or not _sandbox_manager.available:
        return None
    target = org_id if org_id is not None else _get_current_org()
    if target is None:
        return None
    return _sandbox_manager.get_backend(target)


class _OrgScopedSandboxBackendProxy:
    """Proxy that delegates OpensandboxBackend calls to the current org's backend.

    The current org is read from thread-local (set by each request helper).
    When no backend is available, it falls back to a StateBackend-style no-op
    so that agent invocation does not crash outright — behaviour matches the
    existing "degraded mode" semantics.

    Registered as a virtual subclass of ``SandboxBackendProtocol`` so that
    ``CompositeBackend.execute()`` passes its ``isinstance(default,
    SandboxBackendProtocol)`` check.  ``execute`` and ``id`` are implemented
    explicitly (not via ``__getattr__``) because
    ``execute_accepts_timeout(type(proxy))`` inspects the class-level
    ``execute`` signature with ``inspect.signature`` — ``__getattr__`` only
    fires on *instance* attribute access, not class-level introspection.
    """

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

    # ---- Explicitly declared methods required by SandboxBackendProtocol ----
    # These must exist as real class attributes so that:
    #   1. isinstance(proxy, SandboxBackendProtocol) → True  (via register)
    #   2. inspect.signature(type(proxy).execute) works      (for execute_accepts_timeout)
    #   3. The call routes to the correct org's sandbox at runtime

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

    # Forward all other attribute access (read, write, edit, ls, grep, glob,
    # upload_files, download_files, delete, etc.) to the resolved backend.
    def __getattr__(self, name):
        underlying = self._resolve()
        return getattr(underlying, name)


# Register as a virtual subclass so CompositeBackend.execute() recognises the
# proxy as a SandboxBackendProtocol without inheriting all its method stubs
# (which would shadow __getattr__ with NotImplementedError-raising defaults).
SandboxBackendProtocol.register(_OrgScopedSandboxBackendProxy)


# ---- Data models -------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default-thread"
    user_id: str = "local-user"
    org_id: str = "default-org"


class MessageItem(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    reasoning: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    thread_id: str
    messages: List[MessageItem]


class ThreadInfo(BaseModel):
    thread_id: str
    created_at: Optional[str] = None
    last_message: Optional[str] = None


class FileAnalysisResponse(BaseModel):
    file_id: str
    filename: str
    file_type: str
    row_count: int
    columns: List[str]
    preview: str
    analysis_type: str
    code: Optional[str] = None


class ChatWithFileRequest(BaseModel):
    message: str
    thread_id: str = "default-thread"
    user_id: str = "local-user"
    org_id: str = "default-org"
    file_id: Optional[str] = None


def _ensure_user_habits(store: PostgresStore, user_id: str):
    """Ensure /memories/habits.md exists for the given user.

    Each user gets their own habits.md stored under the namespace
    (user_id,) — matching user_namespace() in namespace_router.py —
    so different users' habits are fully isolated in the store.
    """
    HABITS_KEY = "/memories/habits.md"
    NAMESPACE = (user_id,)

    existing = store.get(NAMESPACE, HABITS_KEY)
    if existing:
        return

    initial_content = """# 用户习惯记录

_此文件由 Agent 自动维护，持续记录用户的偏好、习惯和特点。_

## 编程语言与技术偏好
<!-- 用户使用的编程语言、框架、工具等偏好 -->

## 工作习惯
<!-- 用户的工作时间、工作流程、常用工具等 -->

## 沟通与输出偏好
<!-- 用户的语言风格、输出格式、详细程度等偏好 -->

## 其他特征
<!-- 用户的特殊习惯、需求、注意事项等 -->

---

_记录格式：_  
_## [类别名称]_  
_- [具体习惯描述] (来源：YYYY-MM-DD 对话)_
"""

    store.put(NAMESPACE, HABITS_KEY, {
        "content": initial_content,
        "encoding": "utf-8",
    })
    logger.info(f"Created /memories/habits.md for user: {user_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _checkpointer, _agent, _sandbox_manager
    logger.info("Initializing database connections...")

    stack = ExitStack()
    try:
        _store = stack.enter_context(PostgresStore.from_conn_string(DB_URI))
        _checkpointer = stack.enter_context(PostgresSaver.from_conn_string(DB_URI))

        _store.setup()
        _checkpointer.setup()

        # Initialize per-org sandbox manager (optional; degraded mode on failure)
        _sandbox_manager = SandboxManager(SANDBOX_IMAGE)
        _sandbox_manager.start()

        # The agent gets a proxy backend that routes each tool-call to the
        # correct org's sandbox via thread-local org_id context.
        default_backend: Any
        if _sandbox_manager.available:
            default_backend = _OrgScopedSandboxBackendProxy(_sandbox_manager)
            logger.info("Sandbox manager initialized (per-org container, 10-min idle recycle).")
        else:
            default_backend = StateBackend()
            logger.warning("Sandbox unavailable, running in degraded mode (no code execution).")

        HABIT_SYSTEM_PROMPT = """你是一个智能助手，可以通过持续对话学习和记录用户的习惯与偏好。

## 关于用户习惯记录

在对话过程中，你应当主动观察用户的行为、言论和选择，识别以下类型的信息：

### 应该记录的内容：
- **编程语言与技术偏好**：用户常用的编程语言、框架、工具、代码风格等
- **工作习惯**：用户的工作流程、时间安排、常用工具、偏好的方法论等
- **沟通与输出偏好**：用户偏好的语言（中文/英文）、输出详细程度、格式要求、回复风格等
- **其他特征**：用户的特殊需求、注意事项、习惯、个人特点等

### 记录规则：
1. 使用 `edit_file` 工具将识别到的习惯追加到 `/memories/habits.md` 的对应分类下
2. 记录格式为：`- [具体习惯描述] (来源：YYYY-MM-DD 对话)`
3. 每次只记录新发现的习惯，不要重复已有内容
4. 如果用户明确要求"记住"某事，务必记录
5. 每个用户的习惯记录是相互隔离的，你只需关注当前对话用户

### 不应记录的内容：
- 临时性信息（如"我今天迟到了"、"帮我订个会议室"）
- 一次性任务请求
- 简单问答的答案
- 敏感信息（密码、密钥、身份证号等）
- 过期或无意义的闲聊

## PPT 制作能力（ynlczf-pptx-generator 技能）

你拥有 `ynlczf-pptx-generator` 技能，超轻量化生成极简 PPT 并导出 PPTX。仅依赖 python-pptx，无动画/渐变/复杂图表，适合快速汇报、课堂分享、短演示。
当用户要求"做PPT/制作演示文稿/生成PPT/create PPT/做幻灯片/极简PPT/轻量化汇报PPT/导出pptx简报"等时，按以下约定执行：

1. 先用 `read_file` 读取 `/skills/ynlczf-pptx-generator/SKILL.md`（`limit=1000`）了解硬性约束与页面结构。
2. **两种使用方式**（任选其一，推荐方式一）：
   - **方式一（推荐）**：直接用 `execute` 运行沙箱中的脚本：
     ```
     python3 /skills/ynlczf-pptx-generator/generate_pptx.py \
       --topic "主题" \
       --points "要点1,要点2,要点3" \
       --page-num 5 \
       --output /workspace/light_presentation.pptx
     ```
     可选参数：`--data "10,25,30,15"`（极简柱状图数据）、`--json`（以JSON输出结果）
   - **方式二**：自行编写 Python 代码，参考 SKILL.md 中的 `generate_light_pptx` 函数代码，用 `write_file` 写入后再 `execute` 执行。
3. **关键约定**：
   - `SKILL_DIR` = `/skills/ynlczf-pptx-generator`
   - 读取技能文档用 `read_file`（路径以 `/skills/ynlczf-pptx-generator/` 开头）
   - 运行脚本用 `execute` 工具（当前组织沙箱，Linux 环境，`python3` 命令）
   - 核心依赖 `python-pptx` 已预装到沙箱
   - 输出文件保存到沙箱 `/workspace/` 目录下（绝对路径）
4. 生成 PPTX 后，将沙箱中 pptx 的绝对路径填入下载链接格式（见每条消息末尾的【运行环境】），把可点击的下载链接返回给用户。
5. 如遇到依赖缺失，可用 `execute` 运行 `python3 -m pip install python-pptx` 补装后重试。
6. 若沙箱刚创建、脚本尚未预装完成导致首次执行失败，等待几秒后重试即可。
"""

        _agent = create_deep_agent(
            model="deepseek:deepseek-v4-flash",
            context_schema=MemoryContext,
            system_prompt=HABIT_SYSTEM_PROMPT,
            memory=["/memories/AGENTS.md", "/memories/habits.md"],
            skills=["/skills/"],
            checkpointer=_checkpointer,
            backend=CompositeBackend(
                default=default_backend,
                routes={
                    "/memories/": StoreBackend(
                        namespace=user_namespace,
                        store=_store,
                    ),
                    "/skills/": FilesystemBackend(
                        root_dir=str(LOCAL_SKILLS_DIR),
                        virtual_mode=True,
                    ),
                },
            ),
        )

        logger.info("Application started successfully.")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize: {e}", exc_info=True)
        stack.close()
        if _sandbox_manager:
            try:
                _sandbox_manager.stop()
            except Exception:
                pass
        raise

    logger.info("Shutting down...")
    stack.close()
    if _sandbox_manager:
        try:
            _sandbox_manager.stop()
        except Exception as e:
            logger.error(f"Error stopping sandbox manager: {e}")


app = FastAPI(title="AI Chat Web", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_reasoning_and_content(message: Any) -> Tuple[str, Optional[str]]:
    """Split an AIMessage into (answer_content, reasoning_content).

    Reasoning may come from:
      1. DeepSeek/OpenAI compatible models: `additional_kwargs['reasoning_content']`
      2. Inline `<think>...</think>` tags emitted by some reasoning models.

    When reasoning is found inline, it is removed from the returned content so
    that the frontend can render thinking and answer as separate sections.
    For non-AIMessage inputs reasoning is always None.
    """
    if not isinstance(message, AIMessage):
        content = str(getattr(message, "content", "") or "")
        return content, None

    raw_content = str(message.content or "")
    reasoning: Optional[str] = None

    # 1. Structured reasoning from additional_kwargs (DeepSeek, OpenAI o-series)
    extra = message.additional_kwargs or {}
    rc = extra.get("reasoning_content") or extra.get("reasoning")
    if isinstance(rc, str) and rc.strip():
        reasoning = rc.strip()

    # 2. Inline <think>...</think> tags emitted by some reasoning models
    think_match = re.search(r"<think>(.*?)</think>", raw_content, flags=re.DOTALL | re.IGNORECASE)
    if think_match:
        inline_reasoning = think_match.group(1).strip()
        if inline_reasoning:
            reasoning = (reasoning + "\n\n" + inline_reasoning) if reasoning else inline_reasoning
        # Remove the think block from the displayed answer content
        raw_content = (raw_content[:think_match.start()] + raw_content[think_match.end():]).strip()

    return raw_content.strip(), reasoning


def _get_messages_from_state(state: Dict[str, Any]) -> List[MessageItem]:
    messages = state.get("messages", [])
    result = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            role = "assistant" if isinstance(msg, AIMessage) else "user" if isinstance(msg, HumanMessage) else "system"
            content, reasoning = _extract_reasoning_and_content(msg) if isinstance(msg, AIMessage) else (str(msg.content), None)
            result.append(MessageItem(
                role=role,
                content=content,
                reasoning=reasoning,
            ))
        elif isinstance(msg, dict):
            result.append(MessageItem(
                role=msg.get("role", "unknown"),
                content=str(msg.get("content", "")),
                reasoning=msg.get("reasoning"),
            ))
    return result


# ---- Synchronous helper functions (run in thread pool) ----

def _build_env_footer(org_id: str) -> str:
    """Per-request environment footer appended to user messages.

    Gives the agent the current org_id (needed to build a PPT download URL)
    and the sandbox working directory. Kept short to avoid bloating every
    turn.
    """
    from urllib.parse import quote
    return (
        f"\n\n【运行环境】org_id={org_id}；沙箱工作目录 /workspace；"
        f"PPT skill SKILL_DIR=/skills/ynlczf-pptx-generator（脚本与 python-pptx 依赖已预装到沙箱）。"
        f"生成 PPT 后，把沙箱中 pptx 的绝对路径填入此格式给出下载链接："
        f"/api/sandbox/download?org_id={quote(org_id)}&path=<沙箱中pptx绝对路径>"
    )


def _sync_chat(message: str, thread_id: str, user_id: str, org_id: str) -> Dict[str, Any]:
    try:
        _set_current_org(org_id)
        _ensure_user_habits(_store, user_id)
        logger.info(f"Processing chat: thread_id={thread_id}, org_id={org_id}, message={message[:50]}...")
        config = RunnableConfig(configurable={"thread_id": thread_id})
        context = MemoryContext(user_id=user_id, org_id=org_id)
        full_message = message + _build_env_footer(org_id)
        res = _agent.invoke(
            {"messages": [{"role": "user", "content": full_message}]},
            context=context,
            config=config,
        )
        messages = _get_messages_from_state(res)
        last_message = messages[-1] if messages else None
        reply = last_message.content if last_message else ""
        reasoning = last_message.reasoning if last_message else None
        logger.info(f"Chat completed: thread_id={thread_id}, reply_length={len(reply)}")
        return {
            "status": "ok",
            "thread_id": thread_id,
            "reply": reply,
            "reasoning": reasoning,
            "messages": [m.model_dump() for m in messages],
        }
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return {
            "status": "error",
            "thread_id": thread_id,
            "reply": f"Error: {str(e)}",
            "reasoning": None,
            "messages": [],
        }
    finally:
        _set_current_org(None)


def _sync_get_history(thread_id: str) -> ChatHistoryResponse:
    try:
        config = {"configurable": {"thread_id": thread_id}}
        # deepagents stores messages in a DeltaChannel: the raw checkpoint's
        # channel_values does NOT contain a `messages` key (only deltas live in
        # the write log). `_agent.get_state` replays those deltas and returns a
        # fully reconstructed state, so it is the only correct way to read
        # conversation history here.
        snapshot = _agent.get_state(config)
        messages: List[MessageItem] = []
        if snapshot and snapshot.values:
            messages = _get_messages_from_state(snapshot.values)
        return ChatHistoryResponse(thread_id=thread_id, messages=messages)
    except Exception as e:
        logger.error(f"Get history error: {e}", exc_info=True)
        return ChatHistoryResponse(thread_id=thread_id, messages=[])


def _parse_thread_owner(thread_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (org_id, user_id) encoded in a thread_id.

    Thread IDs generated by the frontend follow the format:
        {org_id}__{user_id}__thread-{timestamp}-{random}

    Older thread IDs (format: thread-{timestamp}-{random}) have no owner
    info and return (None, None).
    """
    parts = thread_id.split("__", 2)
    if len(parts) >= 3:
        return parts[0], parts[1]
    return None, None


def _sync_list_threads(
    org_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[ThreadInfo]:
    """List conversation threads, optionally filtered by org_id and user_id.

    - ``org_id``: when provided, only threads belonging to this org are
      returned (mandatory filter for per-org isolation).
    - ``user_id``: when provided (used together with org_id), only threads
      created by this user are returned.
    Threads whose IDs don't encode owner info (legacy format) are excluded
    whenever a filter is active.
    """
    try:
        threads: List[ThreadInfo] = []
        # Use limit to avoid scanning all checkpoints
        tuples = list(_checkpointer.list(None, limit=50))
        for checkpoint_tuple in tuples:
            config = getattr(checkpoint_tuple, "config", {}) or {}
            configurable = config.get("configurable", {}) or {}
            thread_id = configurable.get("thread_id") or "unknown"
            if not thread_id or thread_id == "unknown":
                continue
            if any(t.thread_id == thread_id for t in threads):
                continue

            # --- Owner-based filtering ---
            thread_org, thread_user = _parse_thread_owner(thread_id)
            if org_id is not None:
                if thread_org is None or thread_org != org_id:
                    continue
            if user_id is not None:
                if thread_user is None or thread_user != user_id:
                    continue

            info = ThreadInfo(thread_id=thread_id)
            try:
                snapshot = _agent.get_state({"configurable": {"thread_id": thread_id}})
                if snapshot and snapshot.values:
                    msgs = _get_messages_from_state(snapshot.values)
                    if msgs:
                        last_user_msg = next((m for m in reversed(msgs) if m.role == "user"), None)
                        info.last_message = last_user_msg.content if last_user_msg else (msgs[-1].content[:50] if msgs else None)
            except Exception:
                pass
            threads.append(info)
        return threads
    except Exception as e:
        logger.error(f"List threads error: {e}", exc_info=True)
        return []


def _sync_delete_thread(thread_id: str):
    _checkpointer.delete_thread({"configurable": {"thread_id": thread_id}})


def _sync_upload_file(file_id: str, filename: str, content: bytes, org_id: str) -> FileAnalysisResponse:
    """Parse and upload a file to sandbox, returning analysis metadata.

    The file is uploaded into the sandbox container belonging to `org_id`.
    """
    try:
        _set_current_org(org_id)
        parsed = FileHandler.parse(filename, content)
        _uploaded_files[file_id] = parsed
        _file_org_map[file_id] = org_id
        _file_content_cache[file_id] = content

        sandbox_filename = f"{file_id}_{filename}"
        sandbox_path = f"/workspace/{sandbox_filename}"
        org_backend = _get_org_backend(org_id)
        if org_backend is not None:
            org_backend.upload_files([(sandbox_path, content)])
            logger.info(f"Uploaded file to sandbox (org={org_id}): {sandbox_path}")

        # When sandbox is unavailable, force LLM direct analysis for all files
        if org_backend is None:
            analysis_type = "llm_direct"
            if not parsed.is_small:
                truncated = parsed.preview
                parsed.is_small = True
                parsed.full_content = truncated
            code = None
        else:
            analysis_type = "llm_direct" if parsed.is_small else "code_execution"
            code = None
            if not parsed.is_small:
                code = FileHandler.generate_code_for_large_file(
                    sandbox_filename=sandbox_filename,
                    file_type=parsed.file_type,
                    columns=parsed.columns,
                )

        return FileAnalysisResponse(
            file_id=file_id,
            filename=parsed.filename,
            file_type=parsed.file_type,
            row_count=parsed.row_count,
            columns=parsed.columns,
            preview=parsed.preview,
            analysis_type=analysis_type,
            code=code,
        )
    except Exception as e:
        logger.error(f"File upload error: {e}", exc_info=True)
        raise
    finally:
        _set_current_org(None)


def _sync_chat_with_file(message: str, thread_id: str, user_id: str, org_id: str, file_id: Optional[str]) -> Dict[str, Any]:
    """Chat with file context injected into the message."""
    try:
        _ensure_user_habits(_store, user_id)
        file_context = ""

        if file_id and file_id in _uploaded_files:
            parsed = _uploaded_files[file_id]

            # Use the org_id recorded at upload time — the file lives in that
            # org's sandbox.  If the current request's org_id differs, log a
            # warning so the mismatch is visible during debugging.
            file_org_id = _file_org_map.get(file_id, org_id)
            if file_org_id != org_id:
                logger.warning(
                    f"File org mismatch: file_id={file_id} was uploaded to "
                    f"org={file_org_id} but current request org={org_id}. "
                    f"Using file's org for sandbox access."
                )

            # Set thread-local so the agent's tool calls (execute / write_file)
            # route to the SAME sandbox that holds the file.
            _set_current_org(file_org_id)
            org_backend = _get_org_backend(file_org_id)

            if parsed.is_small and parsed.full_content:
                file_context = f"\n\n【已上传文件内容】\n{parsed.full_content}\n\n请基于以上文件内容回答问题。"
            elif org_backend is not None:
                # Large file — the sandbox may have been recycled since upload
                # (idle > 10 min). Re-upload the file to make sure it exists.
                sandbox_filename = f"{file_id}_{parsed.filename}"
                sandbox_path = f"/workspace/{sandbox_filename}"
                cached_content = _file_content_cache.get(file_id)
                if cached_content is not None:
                    try:
                        org_backend.upload_files([(sandbox_path, cached_content)])
                        logger.info(
                            f"Re-uploaded file to sandbox (org={file_org_id}): "
                            f"{sandbox_path} (ensures file exists after possible recycle)"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to re-upload file to sandbox: {e}")

                file_context = (
                    f"\n\n【已上传文件信息】\n"
                    f"文件名: {parsed.filename}\n"
                    f"文件类型: {parsed.file_type}\n"
                    f"文件在沙箱中的路径: {sandbox_path}\n"
                    f"总行数: {parsed.row_count}\n"
                    f"列名: {parsed.columns}\n\n"
                    f"你可以使用 execute 工具在沙箱中运行命令来分析此文件。\n"
                    f"沙箱环境为 Python 3.12，**没有安装 pandas/numpy/pip**，请使用 Python 内置的 csv 模块。\n"
                    f"示例步骤：\n"
                    f"1. 用 write_file 工具将分析代码写入 /workspace/analyze.py\n"
                    f"2. 用 execute 工具运行 `python3 /workspace/analyze.py`\n"
                    f"代码示例:\n"
                    f"```python\n"
                    f"import csv, json\n"
                    f"with open('{sandbox_path}', 'r', encoding='utf-8') as f:\n"
                    f"    reader = csv.DictReader(f)\n"
                    f"    rows = list(reader)\n"
                    f"# 根据用户问题编写统计逻辑\n"
                    f"print(json.dumps({{'count': len(rows)}}, ensure_ascii=False))\n"
                    f"```\n"
                    f"请根据用户的具体问题编写对应的代码并执行，基于执行结果回答问题。\n\n"
                    f"【文件预览】\n{parsed.preview}"
                )
            else:
                file_context = (
                    f"\n\n【已上传文件信息】\n"
                    f"文件名: {parsed.filename}\n"
                    f"行数: {parsed.row_count}\n"
                    f"列名: {parsed.columns}\n"
                    f"由于文件较大且沙箱不可用，只能提供预览数据。\n\n"
                    f"【文件预览】\n{parsed.preview}"
                )
        else:
            # No file — still set thread-local for normal chat tool calls
            _set_current_org(org_id)

        full_message = message + file_context + _build_env_footer(org_id)

        effective_org = _get_current_org()
        logger.info(
            f"Processing chat with file: thread_id={thread_id}, org_id={org_id}, "
            f"file_id={file_id}, sandbox_org={effective_org}"
        )
        config = RunnableConfig(configurable={"thread_id": thread_id})
        context = MemoryContext(user_id=user_id, org_id=org_id)
        res = _agent.invoke(
            {"messages": [{"role": "user", "content": full_message}]},
            context=context,
            config=config,
        )
        messages = _get_messages_from_state(res)
        last_message = messages[-1] if messages else None
        reply = last_message.content if last_message else ""
        reasoning = last_message.reasoning if last_message else None
        logger.info(f"Chat completed: thread_id={thread_id}, reply_length={len(reply)}")
        return {
            "status": "ok",
            "thread_id": thread_id,
            "reply": reply,
            "reasoning": reasoning,
            "messages": [m.model_dump() for m in messages],
        }
    except Exception as e:
        logger.error(f"Chat with file error: {e}", exc_info=True)
        return {
            "status": "error",
            "thread_id": thread_id,
            "reply": f"Error: {str(e)}",
            "reasoning": None,
            "messages": [],
        }
    finally:
        _set_current_org(None)


# ---- Routes ----

@app.get("/")
async def root():
    frontend_path = Path(__file__).parent / "static" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    return {"status": "ok", "message": "AI Chat API is running"}


# Handle Vite dev client requests (injected by IDE browser) to prevent reload loops
@app.get("/@vite/client")
async def vite_client():
    return Response(content="", media_type="text/javascript")


@app.post("/api/chat")
async def chat(request: Request):
    if not _agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    try:
        body = await request.body()
        data = json.loads(body) if body else {}
        message = data.get("message", "")
        thread_id = data.get("thread_id", "default-thread")
        user_id = data.get("user_id", "local-user")
        org_id = data.get("org_id", "default-org")

        # Run agent in a thread with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(_sync_chat, message, thread_id, user_id, org_id),
            timeout=120.0
        )
        
        # Check if the result has an error status
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("reply", "Unknown error"))
        
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="请求超时，请稍后重试")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/files/upload", response_model=FileAnalysisResponse)
async def upload_file(file: UploadFile = File(...), org_id: str = Form("default-org")):
    """Upload and parse a file for analysis.

    The file is uploaded into the sandbox container belonging to `org_id`.
    Clients should send `org_id` as a multipart form field alongside the file.
    """
    if not _agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    try:
        content = await file.read()
        file_id = f"file_{asyncio.get_event_loop().time():.0f}"

        result = await asyncio.wait_for(
            asyncio.to_thread(_sync_upload_file, file_id, file.filename, content, org_id),
            timeout=30.0
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="文件处理超时")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat-with-file")
async def chat_with_file(request: Request):
    """Chat with file context injected."""
    if not _agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    try:
        body = await request.body()
        data = json.loads(body) if body else {}
        message = data.get("message", "")
        thread_id = data.get("thread_id", "default-thread")
        user_id = data.get("user_id", "local-user")
        org_id = data.get("org_id", "default-org")
        file_id = data.get("file_id")

        result = await asyncio.wait_for(
            asyncio.to_thread(_sync_chat_with_file, message, thread_id, user_id, org_id, file_id),
            timeout=180.0
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("reply", "Unknown error"))

        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="请求超时，请稍后重试")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _sync_execute_code(code: str, timeout: int, org_id: str) -> Dict[str, Any]:
    """Execute code in org's sandbox (runs in thread pool)."""
    try:
        _set_current_org(org_id)
        backend = _get_org_backend(org_id)
        if backend is None:
            raise RuntimeError("沙箱服务不可用，或当前组织没有可用的沙箱容器。")
        result = backend.execute(code, timeout=timeout)
        return {
            "status": "ok" if result.exit_code == 0 else "error",
            "output": result.output,
            "exit_code": result.exit_code,
        }
    finally:
        _set_current_org(None)


@app.post("/api/sandbox/execute")
async def execute_code(request: Request):
    """Execute code in the sandbox environment belonging to `org_id`."""
    if not _sandbox_available():
        raise HTTPException(status_code=503, detail="沙箱服务不可用。请先启动 OpenSandbox server，或将文件分析模式切换为直接分析。")
    try:
        body = await request.body()
        data = json.loads(body) if body else {}
        code = data.get("code", "")
        timeout = data.get("timeout", 60)
        org_id = data.get("org_id", "default-org")

        result = await asyncio.wait_for(
            asyncio.to_thread(_sync_execute_code, code, timeout, org_id),
            timeout=timeout + 10,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="代码执行超时")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _sync_download_sandbox_file(org_id: str, path: str) -> Optional[bytes]:
    """Download a file from the org's sandbox (runs in thread pool)."""
    try:
        _set_current_org(org_id)
        backend = _get_org_backend(org_id)
        if backend is None:
            logger.warning(f"Download: no backend for org={org_id}, path={path}")
            return None
        resp = backend.download_files([path])[0]
        if resp.error:
            logger.warning(f"Download: sandbox returned error for path={path}: {resp.error}")
            return None
        if resp.content is None:
            logger.warning(f"Download: sandbox returned None content for path={path}")
            return None
        logger.info(f"Download: success, org={org_id}, path={path}, size={len(resp.content)} bytes")
        return resp.content
    except Exception as e:
        logger.error(f"Download: exception for org={org_id}, path={path}: {e}", exc_info=True)
        return None
    finally:
        _set_current_org(None)


@app.get("/api/sandbox/download")
async def sandbox_download(org_id: str, path: str):
    """Download a generated artifact (e.g. PPTX) from the org's sandbox.

    The agent constructs a link to this endpoint after exporting a PPTX,
    filling in the org_id and the absolute sandbox path of the .pptx file.
    """
    if not _sandbox_available():
        raise HTTPException(status_code=503, detail="沙箱服务不可用。")
    if not path:
        raise HTTPException(status_code=400, detail="path 参数不能为空。")
    try:
        content = await asyncio.wait_for(
            asyncio.to_thread(_sync_download_sandbox_file, org_id, path),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="下载超时")
    if content is None:
        raise HTTPException(status_code=404, detail="文件不存在或无法从沙箱下载。")
    filename = Path(path).name or "download.pptx"
    if filename.lower().endswith(".pptx"):
        media = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    else:
        media = "application/octet-stream"
    return Response(
        content=content,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/files")
async def list_uploaded_files():
    """List all uploaded files."""
    files = []
    for file_id, parsed in _uploaded_files.items():
        files.append({
            "file_id": file_id,
            "filename": parsed.filename,
            "file_type": parsed.file_type,
            "row_count": parsed.row_count,
            "columns": parsed.columns,
            "is_small": parsed.is_small,
        })
    return {"files": files}


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str):
    """Delete an uploaded file and its cached metadata."""
    _uploaded_files.pop(file_id, None)
    _file_org_map.pop(file_id, None)
    _file_content_cache.pop(file_id, None)
    return {"status": "ok", "file_id": file_id}


@app.get("/api/history/{thread_id}", response_model=ChatHistoryResponse)
async def get_history(thread_id: str):
    if not _checkpointer:
        raise HTTPException(status_code=500, detail="Checkpointer not initialized")
    try:
        return await asyncio.to_thread(_sync_get_history, thread_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/threads", response_model=List[ThreadInfo])
async def list_threads(org_id: Optional[str] = None, only_mine: bool = False, user_id: Optional[str] = None):
    """List conversation threads filtered by organization.

    - ``org_id``: restricts results to threads owned by this org.
    - ``only_mine``: when true (along with ``user_id``), further restricts
      results to threads created by ``user_id``.
    """
    if not _checkpointer:
        raise HTTPException(status_code=500, detail="Checkpointer not initialized")
    try:
        filter_user_id = user_id if only_mine else None
        return await asyncio.to_thread(_sync_list_threads, org_id, filter_user_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str):
    if not _checkpointer:
        raise HTTPException(status_code=500, detail="Checkpointer not initialized")
    try:
        await asyncio.to_thread(_sync_delete_thread, thread_id)
        return {"status": "ok", "thread_id": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/api/health")
async def health_check():
    sandbox_info: Dict[str, Any] = {
        "available": _sandbox_available(),
    }
    if _sandbox_manager is not None:
        sandbox_info["active_sandboxes"] = _sandbox_manager.get_sandbox_count()
    return {
        "status": "ok",
        "agent": _agent is not None,
        "sandbox": sandbox_info,
    }


@app.get("/api/admin/sandboxes")
async def admin_list_sandboxes():
    """Diagnostic endpoint: list active sandboxes per org (without leaking secrets)."""
    if _sandbox_manager is None:
        return {"available": False, "orgs": []}
    orgs = []
    # Safely iterate — we don't expose entry internals, just org id + idle time
    with _sandbox_manager._global_lock:
        items = list(_sandbox_manager._entries.items())
    for org_id, entry in items:
        with entry.lock:
            idle = entry.idle_seconds
        orgs.append({
            "org_id": org_id,
            "idle_seconds": round(idle, 1),
            "will_recycle_after_seconds": max(0, round(SANDBOX_IDLE_TIMEOUT - idle, 1)),
        })
    return {
        "available": _sandbox_manager.available,
        "orgs": orgs,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088, timeout_keep_alive=30)
