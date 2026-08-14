from dataclasses import dataclass
from pydantic import BaseModel


@dataclass(frozen=True)
class MemoryContext:
    user_id: str = "local-user"
    org_id: str = "default-org"
