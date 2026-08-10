from deepagents import create_deep_agent
from deepagents_opensandbox import OpensandboxBackend
from opensandbox.sync.sandbox import SandboxSync
from dotenv import load_dotenv

load_dotenv()

IMAGE = "opensandbox/code-interpreter:v1.1.0"

# 创建沙箱实例
sandbox = SandboxSync.create(IMAGE)
backend = OpensandboxBackend(sandbox=sandbox)

try:
    agent = create_deep_agent(
        model="deepseek-v4-flash",
        system_prompt="你是一个有帮助的AI助手。",
        backend=backend,
    )

    # 展示沙箱环境的作用
    test_prompt = """查看当前环境可以运行那些代码？是否可以运行java、python、node.js等语言的代码？"""

    res = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": test_prompt
            }
        ]
    })
    print(res["messages"][-1])
except Exception as e:
    print(f"错误: {e}")
finally:
    try:
        sandbox.kill()
        sandbox.close()
    except Exception as e:
        print(f"关闭沙箱时出错: {e}")