import asyncio
import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from .agent_setup import _ensure_user_habits
from .config import SANDBOX_IMAGE
from .context_type import MemoryContext
from .models import (
    ChatRequest, MessageItem, ChatHistoryResponse, ThreadInfo,
    FileAnalysisResponse, ChatWithFileRequest,
)
from .sandbox import (
    _set_current_org, _get_current_org, _sandbox_available,
    _get_org_backend, get_sandbox_manager,
)
from tools.file_handler import FileHandler, ParsedFile

logger = logging.getLogger(__name__)

router = APIRouter()

_store: Optional[PostgresStore] = None
_checkpointer: Optional[PostgresSaver] = None
_agent: Any = None
_uploaded_files: Dict[str, ParsedFile] = {}
_file_org_map: Dict[str, str] = {}
_file_content_cache: Dict[str, bytes] = {}


def set_globals(store: PostgresStore, checkpointer: PostgresSaver, agent: Any):
    global _store, _checkpointer, _agent
    _store = store
    _checkpointer = checkpointer
    _agent = agent


def _extract_reasoning_and_content(message: Any) -> Tuple[str, Optional[str]]:
    if not isinstance(message, AIMessage):
        content = str(getattr(message, "content", "") or "")
        return content, None

    raw_content = str(message.content or "")
    reasoning: Optional[str] = None

    extra = message.additional_kwargs or {}
    rc = extra.get("reasoning_content") or extra.get("reasoning")
    if isinstance(rc, str) and rc.strip():
        reasoning = rc.strip()

    think_match = re.search(r"<think>(.*?)</think>", raw_content, flags=re.DOTALL | re.IGNORECASE)
    if think_match:
        inline_reasoning = think_match.group(1).strip()
        if inline_reasoning:
            reasoning = (reasoning + "\n\n" + inline_reasoning) if reasoning else inline_reasoning
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


def _build_env_footer(org_id: str) -> str:
    return (
        f"\n\n【运行环境】org_id={org_id}；沙箱工作目录 /workspace；"
        f"技能目录 /skills/。"
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
        snapshot = _agent.get_state(config)
        messages: List[MessageItem] = []
        if snapshot and snapshot.values:
            messages = _get_messages_from_state(snapshot.values)
        return ChatHistoryResponse(thread_id=thread_id, messages=messages)
    except Exception as e:
        logger.error(f"Get history error: {e}", exc_info=True)
        return ChatHistoryResponse(thread_id=thread_id, messages=[])


def _parse_thread_owner(thread_id: str) -> Tuple[Optional[str], Optional[str]]:
    parts = thread_id.split("__", 2)
    if len(parts) >= 3:
        return parts[0], parts[1]
    return None, None


def _sync_list_threads(
    org_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[ThreadInfo]:
    try:
        threads: List[ThreadInfo] = []
        tuples = list(_checkpointer.list(None, limit=50))
        for checkpoint_tuple in tuples:
            config = getattr(checkpoint_tuple, "config", {}) or {}
            configurable = config.get("configurable", {}) or {}
            thread_id = configurable.get("thread_id") or "unknown"
            if not thread_id or thread_id == "unknown":
                continue
            if any(t.thread_id == thread_id for t in threads):
                continue

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


def _prepare_message_context(message: str, file_id: Optional[str], org_id: str) -> Tuple[str, str]:
    """Build the full message sent to the agent and resolve the org that owns
    the sandbox. Returns (full_message, execution_org)."""
    file_context = ""
    execution_org = org_id

    if file_id and file_id in _uploaded_files:
        parsed = _uploaded_files[file_id]

        file_org_id = _file_org_map.get(file_id, org_id)
        if file_org_id != org_id:
            logger.warning(
                f"File org mismatch: file_id={file_id} was uploaded to "
                f"org={file_org_id} but current request org={org_id}. "
                f"Using file's org for sandbox access."
            )

        _set_current_org(file_org_id)
        org_backend = _get_org_backend(file_org_id)
        execution_org = file_org_id

        if parsed.is_small and parsed.full_content:
            file_context = f"\n\n【已上传文件内容】\n{parsed.full_content}\n\n请基于以上文件内容回答问题。"
        elif org_backend is not None:
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
        _set_current_org(org_id)

    return message + file_context + _build_env_footer(org_id), execution_org


def _sync_chat_with_file(message: str, thread_id: str, user_id: str, org_id: str, file_id: Optional[str]) -> Dict[str, Any]:
    try:
        _ensure_user_habits(_store, user_id)
        full_message, execution_org = _prepare_message_context(message, file_id, org_id)
        _set_current_org(execution_org)

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


def _sync_execute_code(code: str, timeout: int, org_id: str) -> Dict[str, Any]:
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


def _sync_download_sandbox_file(org_id: str, path: str) -> Optional[bytes]:
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


# ---------------------------------------------------------------------------
# Streaming (SSE) helpers — 基于 DeepAgents/LangGraph 原生 graph.stream()
# ---------------------------------------------------------------------------

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _sse(event_type: str, payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps({'type': event_type, **payload}, ensure_ascii=False)}\n\n"


def _coerce_text(value: Any) -> str:
    """Extract plain text from a message content (str or list of blocks)."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


class _ThinkStreamSplitter:
    """Incrementally split a streaming text stream into reasoning
    (<think>...</think>) and visible content."""

    def __init__(self, on_reasoning, on_content):
        self._on_reasoning = on_reasoning
        self._on_content = on_content
        self._buffer = ""

    def push(self, text: str) -> None:
        self._buffer += text
        self._drain()

    def flush(self) -> None:
        if self._buffer:
            self._on_content(self._buffer)
            self._buffer = ""

    def _drain(self) -> None:
        while True:
            buf = self._buffer
            if "<think" not in buf:
                if buf:
                    self._on_content(buf)
                    self._buffer = ""
                return
            m = THINK_RE.search(buf)
            if m:
                if m.start() > 0:
                    self._on_content(buf[:m.start()])
                reasoning = m.group(1).strip()
                if reasoning:
                    self._on_reasoning(reasoning)
                self._buffer = buf[m.end():]
                continue
            # Unclosed <think at the tail: flush completed content before it,
            # keep the tag onward until it closes.
            last_start = buf.rfind("<think")
            if last_start > 0:
                self._on_content(buf[:last_start])
                self._buffer = buf[last_start:]
                continue
            return


def _extract_tool_calls(update: Any) -> List[str]:
    """Recursively walk a node update (dict/list) for parsed tool calls."""
    names: List[str] = []

    def walk(value: Any):
        if isinstance(value, BaseMessage):
            for tc in getattr(value, "tool_calls", None) or []:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name:
                    names.append(name)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)

    walk(update)
    return names


def _stream_sse(
    message: str,
    thread_id: str,
    user_id: str,
    org_id: str,
    file_id: Optional[str] = None,
) -> Iterator[str]:
    """同步生成器：直接调用 DeepAgents/LangGraph 原生的 graph.stream()，
    逐个产出 SSE 事件字符串。由 Starlette StreamingResponse 在线程池中迭代。"""
    reasoning_parts: List[str] = []
    content_parts: List[str] = []
    seen_tools = set()
    pending_events: List[str] = []

    def on_reasoning(text: str):
        reasoning_parts.append(text)
        pending_events.append(_sse("reasoning_token", {"content": text}))

    def on_content(text: str):
        content_parts.append(text)
        pending_events.append(_sse("token", {"content": text}))

    splitter = _ThinkStreamSplitter(on_reasoning, on_content)

    def drain() -> List[str]:
        if not pending_events:
            return []
        events = pending_events[:]
        pending_events.clear()
        return events

    try:
        _ensure_user_habits(_store, user_id)
        full_message, execution_org = _prepare_message_context(message, file_id, org_id)
        # 防御性设置：保证首个迭代线程的上下文正确（后续线程由端点上下文传播）
        _set_current_org(execution_org)

        logger.info(
            f"Streaming chat: thread_id={thread_id}, org_id={org_id}, "
            f"file_id={file_id}, sandbox_org={_get_current_org()}"
        )
        config = RunnableConfig(configurable={"thread_id": thread_id})
        context = MemoryContext(user_id=user_id, org_id=org_id)

        for mode, data in _agent.stream(
            {"messages": [{"role": "user", "content": full_message}]},
            context=context,
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "updates":
                # 节点完成后的完整更新：从中识别工具调用
                for name in _extract_tool_calls(data):
                    if name and name not in seen_tools:
                        seen_tools.add(name)
                        yield _sse("tool_call", {"name": name})
                continue

            # mode == "messages": (chunk, metadata)
            chunk, _meta = data
            if isinstance(chunk, ToolMessage):
                output = _coerce_text(chunk.content).strip()
                if output:
                    short = output if len(output) <= 300 else output[:300] + "..."
                    yield _sse("tool_result", {"output": short})
                continue
            if not isinstance(chunk, AIMessageChunk):
                continue

            # 流式思考内容（如 deepseek 的 reasoning_content）
            extra = chunk.additional_kwargs or {}
            rc = extra.get("reasoning_content") or extra.get("reasoning")
            rc_text = _coerce_text(rc)
            if rc_text:
                reasoning_parts.append(rc_text)
                yield _sse("reasoning_token", {"content": rc_text})

            # 正文内容（把 <think> 块拆到推理区，正文实时显示）
            text = _coerce_text(chunk.content)
            if text:
                splitter.push(text)
                for ev in drain():
                    yield ev
    except Exception as e:
        logger.error(f"Streaming chat error: thread_id={thread_id}: {e}", exc_info=True)
        yield _sse("error", {"message": str(e)})
        return
    finally:
        _set_current_org(None)

    splitter.flush()
    for ev in drain():
        yield ev

    full_content = "".join(content_parts).strip()
    full_reasoning = "\n\n".join(p for p in reasoning_parts if p.strip()).strip() or None
    logger.info(
        f"Streaming chat completed: thread_id={thread_id}, reply_length={len(full_content)}"
    )
    yield _sse("done", {"reply": full_content, "reasoning": full_reasoning})


@router.post("/chat")
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

        result = await asyncio.wait_for(
            asyncio.to_thread(_sync_chat, message, thread_id, user_id, org_id),
            timeout=120.0
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


@router.post("/files/upload", response_model=FileAnalysisResponse)
async def upload_file(file: UploadFile = File(...), org_id: str = Form("default-org")):
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


@router.post("/chat-with-file")
async def chat_with_file(request: Request):
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


def _stream_headers() -> Dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


@router.post("/chat/stream")
async def chat_stream(request: Request):
    if not _agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    body = await request.body()
    data = json.loads(body) if body else {}
    message = data.get("message", "")
    thread_id = data.get("thread_id", "default-thread")
    user_id = data.get("user_id", "local-user")
    org_id = data.get("org_id", "default-org")
    if not message.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    # 在端点上下文设置 org：Starlette 在线程池迭代同步生成器时，
    # anyio 会把本任务的 contextvars 拷贝到每个工作线程，保证工具路由正确。
    _set_current_org(org_id)
    return StreamingResponse(
        _stream_sse(message, thread_id, user_id, org_id),
        media_type="text/event-stream",
        headers=_stream_headers(),
    )


@router.post("/chat-with-file/stream")
async def chat_with_file_stream(request: Request):
    if not _agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    body = await request.body()
    data = json.loads(body) if body else {}
    message = data.get("message", "")
    thread_id = data.get("thread_id", "default-thread")
    user_id = data.get("user_id", "local-user")
    org_id = data.get("org_id", "default-org")
    file_id = data.get("file_id")
    if not message.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    # 与 _prepare_message_context 保持一致：文件所属组织的沙箱优先
    execution_org = _file_org_map.get(file_id, org_id) if file_id else org_id
    _set_current_org(execution_org)
    return StreamingResponse(
        _stream_sse(message, thread_id, user_id, org_id, file_id),
        media_type="text/event-stream",
        headers=_stream_headers(),
    )


@router.post("/sandbox/execute")
async def execute_code(request: Request):
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


@router.get("/sandbox/download")
async def sandbox_download(org_id: str, path: str):
    from fastapi.responses import Response
    content = await asyncio.to_thread(_sync_download_sandbox_file, org_id, path)
    if content is None:
        raise HTTPException(status_code=404, detail="File not found or sandbox unavailable")
    filename = path.split("/")[-1]
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"}
    )


@router.get("/sandbox/status")
async def sandbox_status():
    manager = get_sandbox_manager()
    if not manager or not manager.available:
        return {"available": False, "active_count": 0}
    return {
        "available": True,
        "active_count": manager.get_sandbox_count(),
    }


@router.get("/history/{thread_id}", response_model=ChatHistoryResponse)
async def get_history(thread_id: str):
    return await asyncio.to_thread(_sync_get_history, thread_id)


@router.get("/threads", response_model=List[ThreadInfo])
async def list_threads(org_id: Optional[str] = None, user_id: Optional[str] = None):
    return await asyncio.to_thread(_sync_list_threads, org_id, user_id)


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    await asyncio.to_thread(_sync_delete_thread, thread_id)
    return {"status": "ok"}
