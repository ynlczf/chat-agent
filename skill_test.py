from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv

load_dotenv()

backend = FilesystemBackend(root_dir="./", virtual_mode=True)

agent = create_deep_agent(
    model="deepseek-v4-flash",
    backend=backend,
    skills=["/skills/"]
)

res = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "我是谁"
        }
    ]
})
print(res["messages"][-1])
