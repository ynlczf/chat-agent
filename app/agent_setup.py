import logging
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StoreBackend, StateBackend, FilesystemBackend
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver

from .config import MODEL_NAME, LOCAL_SKILLS_DIR
from .context_type import MemoryContext
from .namespace_router import user_namespace
from .sandbox import _OrgScopedSandboxBackendProxy, get_sandbox_manager, _sandbox_available

logger = logging.getLogger(__name__)

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


def _ensure_user_habits(store: PostgresStore, user_id: str):
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


def create_agent(checkpointer: PostgresSaver, store: PostgresStore) -> Any:
    manager = get_sandbox_manager()
    if manager and manager.available:
        default_backend = _OrgScopedSandboxBackendProxy(manager)
        logger.info("Sandbox manager initialized (per-org container, 10-min idle recycle).")
    else:
        default_backend = StateBackend()
        logger.warning("Sandbox unavailable, running in degraded mode (no code execution).")

    agent = create_deep_agent(
        model=MODEL_NAME,
        context_schema=MemoryContext,
        system_prompt=HABIT_SYSTEM_PROMPT,
        memory=["/memories/AGENTS.md", "/memories/habits.md"],
        skills=["/skills/"],
        checkpointer=checkpointer,
        backend=CompositeBackend(
            default=default_backend,
            routes={
                "/memories/": StoreBackend(
                    namespace=user_namespace,
                    store=store,
                ),
                "/skills/": FilesystemBackend(
                    root_dir=str(LOCAL_SKILLS_DIR),
                    virtual_mode=True,
                ),
            },
        ),
    )
    return agent
