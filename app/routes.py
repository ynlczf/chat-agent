import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
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


def _sync_chat_with_file(message: str, thread_id: str, user_id: str, org_id: str, file_id: Optional[str]) -> Dict[str, Any]:
    try:
        _ensure_user_habits(_store, user_id)
        file_context = ""

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
