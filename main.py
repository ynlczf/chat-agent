from contextlib import asynccontextmanager, ExitStack
import re
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import os
import asyncio
import json
import uvicorn
from pathlib import Path
import logging

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StoreBackend, StateBackend
from deepagents_opensandbox import OpensandboxBackend
from opensandbox.sync.sandbox import SandboxSync
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver

from context_type import MemoryContext
from namespace_router import assistant_namespace
from tools.file_handler import FileHandler, ParsedFile

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

DB_URI = os.getenv("DB_URI", "postgresql://postgres:root@127.0.0.1:5432/testdb")

SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "opensandbox/code-interpreter:v1.1.0")

# Global references for resources
_store = None
_checkpointer = None
_agent = None
_sandbox_backend = None
_uploaded_files: Dict[str, ParsedFile] = {}


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _checkpointer, _agent, _sandbox_backend
    logger.info("Initializing database connections...")

    stack = ExitStack()
    _sandbox = None
    try:
        _store = stack.enter_context(PostgresStore.from_conn_string(DB_URI))
        _checkpointer = stack.enter_context(PostgresSaver.from_conn_string(DB_URI))

        _store.setup()
        _checkpointer.setup()

        # Sandbox initialization is optional — app can run without it
        # (file code-execution feature will be disabled, LLM direct analysis still works)
        default_backend = StateBackend()
        try:
            logger.info("Initializing sandbox backend...")
            _sandbox = SandboxSync.create(SANDBOX_IMAGE)
            _sandbox_backend = OpensandboxBackend(sandbox=_sandbox)
            default_backend = _sandbox_backend
            logger.info("Sandbox backend initialized successfully.")
        except Exception as e:
            logger.warning(f"Sandbox unavailable, running in degraded mode (no code execution): {e}")
            _sandbox_backend = None
            _sandbox = None

        _agent = create_deep_agent(
            model="deepseek:deepseek-v4-flash",
            context_schema=MemoryContext,
            memory=["/memories/AGENTS.md"],
            checkpointer=_checkpointer,
            backend=CompositeBackend(
                default=default_backend,
                routes={
                    "/memories/": StoreBackend(
                        namespace=assistant_namespace,
                        store=_store,
                    ),
                },
            ),
        )

        logger.info("Application started successfully.")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        stack.close()
        if _sandbox:
            try:
                _sandbox.kill()
                _sandbox.close()
            except Exception:
                pass
        raise

    logger.info("Shutting down...")
    stack.close()
    if _sandbox:
        try:
            _sandbox.kill()
            _sandbox.close()
        except Exception as e:
            logger.error(f"Error killing sandbox: {e}")


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

def _sync_chat(message: str, thread_id: str, user_id: str, org_id: str) -> Dict[str, Any]:
    try:
        logger.info(f"Processing chat: thread_id={thread_id}, message={message[:50]}...")
        config = RunnableConfig(configurable={"thread_id": thread_id})
        context = MemoryContext(user_id=user_id, org_id=org_id)
        res = _agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
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


def _sync_list_threads() -> List[ThreadInfo]:
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


def _sync_upload_file(file_id: str, filename: str, content: bytes) -> FileAnalysisResponse:
    """Parse and upload a file to sandbox, returning analysis metadata."""
    try:
        parsed = FileHandler.parse(filename, content)
        _uploaded_files[file_id] = parsed

        sandbox_filename = f"{file_id}_{filename}"
        sandbox_path = f"/workspace/{sandbox_filename}"
        if _sandbox_backend:
            _sandbox_backend.upload_files([(sandbox_path, content)])
            logger.info(f"Uploaded file to sandbox: {sandbox_path}")

        # When sandbox is unavailable, force LLM direct analysis for all files
        if _sandbox_backend is None:
            analysis_type = "llm_direct"
            # For large files without sandbox, include a truncated preview as context
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


def _sync_chat_with_file(message: str, thread_id: str, user_id: str, org_id: str, file_id: Optional[str]) -> Dict[str, Any]:
    """Chat with file context injected into the message."""
    try:
        file_context = ""
        if file_id and file_id in _uploaded_files:
            parsed = _uploaded_files[file_id]
            if parsed.is_small and parsed.full_content:
                file_context = f"\n\n【已上传文件内容】\n{parsed.full_content}\n\n请基于以上文件内容回答问题。"
            elif _sandbox_backend:
                # Large file uploaded to sandbox — tell the agent the file path
                # so it can write and execute its own analysis code (via OpensandboxBackend)
                sandbox_filename = f"{file_id}_{parsed.filename}"
                sandbox_path = f"/workspace/{sandbox_filename}"
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

        full_message = message + file_context

        logger.info(f"Processing chat with file: thread_id={thread_id}, file_id={file_id}")
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
async def upload_file(file: UploadFile = File(...)):
    """Upload and parse a file for analysis."""
    if not _agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    try:
        content = await file.read()
        file_id = f"file_{asyncio.get_event_loop().time():.0f}"

        result = await asyncio.wait_for(
            asyncio.to_thread(_sync_upload_file, file_id, file.filename, content),
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


def _sync_execute_code(code: str, timeout: int) -> Dict[str, Any]:
    """Execute code in sandbox (runs in thread pool)."""
    result = _sandbox_backend.execute(code, timeout=timeout)
    return {
        "status": "ok" if result.exit_code == 0 else "error",
        "output": result.output,
        "exit_code": result.exit_code,
    }


@app.post("/api/sandbox/execute")
async def execute_code(request: Request):
    """Execute code in sandbox environment."""
    if not _sandbox_backend:
        raise HTTPException(status_code=503, detail="沙箱服务不可用。请先启动 OpenSandbox server，或将文件分析模式切换为直接分析。")
    try:
        body = await request.body()
        data = json.loads(body) if body else {}
        code = data.get("code", "")
        timeout = data.get("timeout", 60)

        result = await asyncio.wait_for(
            asyncio.to_thread(_sync_execute_code, code, timeout),
            timeout=timeout + 10,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="代码执行超时")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
    """Delete an uploaded file."""
    if file_id in _uploaded_files:
        del _uploaded_files[file_id]
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
async def list_threads():
    if not _checkpointer:
        raise HTTPException(status_code=500, detail="Checkpointer not initialized")
    try:
        return await asyncio.to_thread(_sync_list_threads)
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
    return {"status": "ok", "agent": _agent is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088, timeout_keep_alive=30)
