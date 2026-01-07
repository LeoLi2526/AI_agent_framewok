# Memo Assistant Agent (LangGraph Integration)

本项目是一个基于大模型和 LangGraph 的智能备忘助手系统。它集成了语音识别、本地备忘提取模型（Qwen）、以及多个 API 模型（如 DeepSeek）来处理对话、用户习惯和突发事件，最终生成智能化的备忘和提醒。

## 项目结构

```
langchain/
├── src/                  # 源代码目录
│   ├── agents/           # 各类 Agent 实现
│   │   ├── audio_agent.py      # 语音识别 Agent
│   │   ├── memo_assistant.py   # 本地 Qwen 备忘助手
│   │   ├── multimodal_agent.py # 多模态突发事件分析
│   │   └── habit_detector.py   # 习惯检测（未使用？）
│   ├── utils/            # 工具类
│   │   ├── audio_devices.py    # 音频设备列表工具
│   │   └── B64PCMPlayer.py     # 音频播放工具
│   ├── prompts/          # 各类 Agent 的 Prompt 模板
│   ├── inputs/           # 输入数据样例
│   └── main.py           # 主程序入口，定义了 LangGraph 工作流
├── README.md             # 项目说明文档
└── requirements.txt      # 项目依赖列表
```

## 功能模块

该系统通过 LangGraph 组织了以下处理节点：

1.  **Audio Agent (`node_audio_agent`)**:
    *   **功能**: 检测麦克风输入，实时将语音转换为文本。
    *   **依赖**: `src/agents/audio_agent.py` 模块，`dashscope` API。
    *   **回退机制**: 如果未检测到麦克风，会自动跳过该节点。

2.  **Local Memo Agent (`node_local_memo`)**:
    *   **功能**: 调用本地部署的 Qwen 模型 (`Qwen3-4B-Instruct-2507`)，从对话内容（包含语音转写的文本）中提取初步的备忘事项。
    *   **核心类**: `MemoAssistant` (在 `src/agents/memo_assistant.py` 中定义)。

3.  **API Alpha Agent (`node_api_alpha`)**:
    *   **功能**: 结合突发事件信息、用户习惯和备忘事项，检测是否有紧急冲突，并生成提醒。
    *   **模型**: 使用 `ChatOpenAI` 接口调用兼容模型（如 DeepSeek-V3）。

4.  **API Beta Agent (`node_api_beta`)**:
    *   **功能**: 专注于检测备忘事项与用户长期行为习惯之间的冲突。
    *   **模型**: 使用 `ChatOpenAI` 接口调用兼容模型。

5.  **Multimodal Agent (`node_multimodal_agent`)**:
    *   **功能**: 分析包含突发事件信息的图片和文本。
    *   **核心类**: `EventDescriptionAgent` (在 `src/agents/multimodal_agent.py` 中定义)。

6.  **API Charlie Agent (`node_api_charlie`)**:
    *   **功能**: 作为最终的整合者，综合 Local Memo 的结果以及 Alpha 和 Beta 的提醒，输出最终的结构化备忘录和建议。
    *   **模型**: DeepSeek-Reasoner (通过 `deepseek_official_llm` 调用)。

## 快速开始

1.  安装依赖:
    ```bash
    pip install -r src/requirements.txt
    ```

2.  配置环境变量:
    在 `src/.env` 文件中配置 API Key:
    ```
    DASHSCOPE_API_KEY=sk-xxx
    DEEPSEEK_API_KEY=sk-xxx
    LANGFUSE_PUBLIC_KEY=pk-lf-xxx
    LANGFUSE_SECRET_KEY=sk-lf-xxx
    LANGFUSE_HOST=https://cloud.langfuse.com
    ```

3.  运行主程序:
    ```bash
    python src/main.py
    ```

## 注意事项

*   **模型路径**: 本地 Qwen 模型路径在 `src/agents/memo_assistant.py` 中配置 (默认为 `/home/leoli/Downloads_for_ai/Models/Qwen3-4B-Instruct-2507`)。
*   **输入文件**: 输入示例文件位于 `src/inputs/` 目录下。

## 环境依赖

请确保已安装 Python 3.11+，并安装以下依赖：

```bash
pip install -r requirements.txt
```

核心依赖包括：
*   `langgraph`, `langchain`, `langchain-openai`: 用于构建 Agent 工作流。
*   `transformers`, `torch`: 用于运行本地大模型。
*   `pyaudio`, `dashscope`: 用于语音识别功能。
*   `langfuse`: 用于链路追踪和监控。

## 配置说明

在项目根目录下创建或修改 `.env` 文件，配置必要的 API Key 和路径：

```env
DASHSCOPE_API_KEY=your_dashscope_key
DEEPSEEK_API_KEY=your_deepseek_key
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=...
```

**注意**: 本地模型的路径在 `memo_assistant.py` 和 `graph_app.py` 中有默认配置（如 `/home/leoli/Downloads_for_ai/Models/Qwen3-4B-Instruct-2507`），请根据实际环境进行修改。

## 运行方法

确保 conda 环境已激活（例如 `conda activate langchain`），然后在终端运行：

```bash
python graph_app.py
```

程序将：
1.  自动检测音频设备，如果有麦克风则进入语音录制模式。
2.  读取 `inputs/` 目录下的文本文件作为上下文输入。
3.  按顺序执行 Graph 中的各个节点。
4.  在控制台输出最终的备忘录结果。

## 语音输入

如果连接了麦克风，程序启动后会尝试进行语音识别。你可以直接说话，语音内容会被实时转写并追加到对话记录中，参与后续的备忘提取流程。
