# Chat Agent

基于 DeepSeek + LangGraph + FastAPI 的智能对话 Agent，支持沙箱代码执行、PPT 生成技能、用户习惯记忆等能力。

## 功能特性

- 🤖 **多轮对话**：基于 LangGraph 的状态化对话
- 🧠 **习惯记忆**：自动记录用户偏好到 PostgresStore
- 🖥️ **沙箱执行**：OpenSandbox 容器化代码执行环境
- 📊 **PPT 生成**：内置极简 PPT 生成技能（python-pptx）
- 📁 **文件处理**：Excel / CSV 智能解析与分析

## 快速开始

### 环境要求

- Python 3.11+
- PostgreSQL
- uv（包管理器）

### 安装

```bash
git clone git@github.com:ynlczf/chat-agent.git
cd chat-agent
uv sync
cp .env.example .env
```

### 启动

```bash
uv run python main.py
```

默认监听 `http://0.0.0.0:8088`

## 项目结构

```
├── app/
├── skills/
├── static/
├── tools/
├── main.py
└── pyproject.toml
```

## License

MIT
