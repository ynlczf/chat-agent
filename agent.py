from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StoreBackend, StateBackend
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver

from context_type import MemoryContext
from namespace_router import user_namespace

load_dotenv()

DB_URI = "postgresql://postgres:root@127.0.0.1:5432/testdb"


def _ensure_user_habits(store: PostgresStore, user_id: str):
    """Ensure /memories/habits.md exists for the given user."""
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
    print(f"Created /memories/habits.md for user: {user_id}")


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
"""

TEST_USER_A = "user-alice"
TEST_USER_B = "user-bob"

with PostgresStore.from_conn_string(DB_URI) as store, \
    PostgresSaver.from_conn_string(DB_URI) as checkpointer:

    checkpointer.setup()
    store.setup()

    agent = create_deep_agent(
        model="deepseek:deepseek-v4-flash",
        context_schema=MemoryContext,
        system_prompt=HABIT_SYSTEM_PROMPT,
        memory=["/memories/AGENTS.md", "/memories/habits.md"],
        backend=CompositeBackend(
            default=StateBackend(),
            routes={
                "/memories/": StoreBackend(
                    namespace=user_namespace,
                    store=store,
                ),
            },
        ),
    )

    # --- User A: Alice ---
    _ensure_user_habits(store, TEST_USER_A)
    res_a = agent.invoke(
        {"messages": [{"role": "user", "content": "我习惯用 Python 处理数据，并且喜欢用简洁的风格输出代码"}]},
        context=MemoryContext(user_id=TEST_USER_A, org_id="org-acme"),
        config=RunnableConfig(configurable={"thread_id": "alice-1"}),
    )
    print(f"[{TEST_USER_A}] Reply:", res_a["messages"][-1].content)

    # --- User B: Bob ---
    _ensure_user_habits(store, TEST_USER_B)
    res_b = agent.invoke(
        {"messages": [{"role": "user", "content": "我喜欢用 Java 开发后端，回复请用英文"}]},
        context=MemoryContext(user_id=TEST_USER_B, org_id="org-acme"),
        config=RunnableConfig(configurable={"thread_id": "bob-1"}),
    )
    print(f"\n[{TEST_USER_B}] Reply:", res_b["messages"][-1].content)

    # --- Verify isolation ---
    print("\n" + "=" * 60)
    print("Verifying user isolation:")
    print("=" * 60)

    habits_a = store.get((TEST_USER_A,), "/memories/habits.md")
    if habits_a:
        print(f"\n--- {TEST_USER_A}'s habits.md ---")
        print(habits_a.value["content"])

    habits_b = store.get((TEST_USER_B,), "/memories/habits.md")
    if habits_b:
        print(f"\n--- {TEST_USER_B}'s habits.md ---")
        print(habits_b.value["content"])
