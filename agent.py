from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StoreBackend, StateBackend
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver

from context_type import MemoryContext
from namespace_router import *

load_dotenv()

DB_URI = "postgresql://postgres:root@127.0.0.1:5432/testdb"

with PostgresStore.from_conn_string(DB_URI) as store, \
    PostgresSaver.from_conn_string(DB_URI) as checkpointer:

    checkpointer.setup()
    store.setup()

    agent = create_deep_agent(
        model="deepseek:deepseek-v4-flash",
        context_schema=MemoryContext,
        memory=["/memories/AGENTS.md"],
        backend=CompositeBackend(
            default=StateBackend(),
            routes={
                "/memories/": StoreBackend(
                    namespace=assistant_namespace,
                    store=store,
                ),
            },
        ),
    )

    res = agent.invoke(
        {"messages": [{"role": "user", "content": "今天你吃了吗"}]},
        context=MemoryContext(user_id="user-1232", org_id="org-acme"),
        config=RunnableConfig(configurable={"thread_id": "1"}),
    )
    print(res["messages"][-1]) 