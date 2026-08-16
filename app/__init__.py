from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from dotenv import load_dotenv

from .agent_setup import create_agent, _ensure_agents_memory, _remove_per_user_agents_memory
from .config import DB_URI, SANDBOX_IMAGE
from .routes import router, set_globals
from .sandbox import init_sandbox_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

_store: Any = None
_checkpointer: Any = None
_agent: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _checkpointer, _agent
    logger.info("Initializing database connections...")

    # Ensure auth tables exist before anything else touches Postgres
    try:
        from .auth import ensure_tables
        ensure_tables()
    except Exception as e:
        logger.warning(f"Auth table init failed (non-fatal): {e}")

    from contextlib import ExitStack
    from langgraph.store.postgres import PostgresStore
    from langgraph.checkpoint.postgres import PostgresSaver

    stack = ExitStack()
    try:
        _store = stack.enter_context(PostgresStore.from_conn_string(DB_URI))
        _checkpointer = stack.enter_context(PostgresSaver.from_conn_string(DB_URI))

        _store.setup()
        _checkpointer.setup()

        init_sandbox_manager(SANDBOX_IMAGE)

        _agent = create_agent(_checkpointer, _store)
        set_globals(_store, _checkpointer, _agent)

        # AGENTS.md 是全局配置：启动时确保唯一一份存在，并清理历史遗留的按用户副本
        try:
            _ensure_agents_memory(_store)
            _remove_per_user_agents_memory(_store)
        except Exception as e:
            logger.warning(f"Failed to seed AGENTS.md at startup: {e}", exc_info=True)

        logger.info("Application started successfully.")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize: {e}", exc_info=True)
        stack.close()
        from .sandbox import get_sandbox_manager
        mgr = get_sandbox_manager()
        if mgr:
            try:
                mgr.stop()
            except Exception:
                pass
        raise

    logger.info("Shutting down...")
    stack.close()
    from .sandbox import get_sandbox_manager
    mgr = get_sandbox_manager()
    if mgr:
        try:
            mgr.stop()
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


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Populate request.state.user from Bearer token on every request.
    Non-fatal: bad/missing token leaves user = None; individual endpoints
    that require auth can check for it."""
    from .auth import inject_user_to_request
    try:
        inject_user_to_request(request)
    except Exception as e:
        logger.debug(f"Auth middleware soft-fail: {e}")
        request.state.user = None
    response = await call_next(request)
    return response


# API routers
app.include_router(router, prefix="/api")

from .auth import router as auth_router
app.include_router(auth_router, prefix="/api")


@app.get("/")
async def root():
    frontend_path = Path(__file__).parent.parent / "static" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    return {"status": "ok", "message": "AI Chat API is running"}


@app.get("/@vite/client")
async def vite_client():
    return Response(content="", media_type="text/javascript")
