from typing import Optional, List

from pydantic import BaseModel


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
