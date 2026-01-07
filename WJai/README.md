# WJai — AI 事件与备忘处理框架

## 项目简介
WJai 是一个用于处理现场多模态突发事件、语音转写与备忘生成的轻量级框架。它结合了多模态事件分析（图片/文本）、实时语音转写（Dashscope）、以及基于 API 或本地模型的备忘生成模块。

## 主要功能
- 多模态突发事件分析：从 `inputs/Emergency_information` 目录读取图片和文本，使用 `EventDescriptionAgent` 生成结构化事件描述。
- 实时/文件语音转写：`audio_agent` 支持多麦克风模式或处理音频文件，生成转写并保存为 JSON。
- 备忘助手：`memo_assistant` 可以使用本地大型模型（如 Qwen 系列）或 API 模型根据对话/转写生成结构化备忘（并进行隐私脱敏）。
- 工作流编排：`src/main.py` 使用 `langgraph.StateGraph` 将以上节点串联为一个端到端流程。

## 目录结构（简要）
- `src/`：主程序目录
  - `main.py`：启动与流程定义（多模态 -> 音频 -> 备忘 -> API 处理 -> 输出）
  - `agents/`：各 agent 实现
    - `audio_agent.py`：麦克风与文件转写逻辑（Dashscope 实时 API）
    - `multimodal_agent.py`：事件分析 Agent（`EventDescriptionAgent`）
    - `memo_assistant.py`：本地/离线备忘生成器（基于 transformers）
    - `multimodal_examples.json`：多模态示例格式（供 Agent 参考）
  - `prompts/`：用于 API 的提示模板（`api_alpha_prompt.txt`、`memo_assistant_system.txt` 等）
  - `utils/`：工具函数（`B64PCMPlayer.py`、`privacy_utils.py` 等）
- `inputs/`：场景输入文件（对话、习惯、示例、突发事件目录）
- `privacy/`：隐私脱敏与日志保存目录（运行时创建）

## 依赖
项目依赖列在 `src/requirements.txt`。建议在虚拟环境中安装：

```bash
conda create -n "your_env_name" python=3.11
pip install -r src/requirements.txt
```

注意：部分依赖（如本地大模型、`dashscope`、`pyaudio`）对系统和硬件有特殊要求。

## 环境变量
- `DASHSCOPE_API_KEY`：Dashscope 实时转写 API Key（用于 `audio_agent` 与多模态 Agent）
- `DEEPSEEK_API_KEY`：DeepSeek API Key（主流程中用于 `api_llm`）
- 其他 LLM API Keys：若使用不同 API，请在 `.env` 或环境变量中配置。

在 `src/main.py` 目录下可放 `.env`，或在系统中导出：

```bash
export DASHSCOPE_API_KEY="your_dashscope_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
```

## 快速运行
1) 运行主流程（端到端）

```bash
python src/main.py
```

说明：`main.py` 会尝试：
- 使用 `multimodal_agent` 处理 `inputs/Emergency_information` 中最新子目录的图片/文本，生成 `inputs/processed_emergency.json`；
- 检测音频输入设备并启动 `audio_agent`（若可用）进行实时转写；
- 使用 API 模型（Alpha/Beta/Charlie）对备忘进行生成与检测。

2) 单独运行音频转写（麦克风）

```bash
python src/agents/audio_agent.py --mode mic
```

或处理音频文件：

```bash
python src/agents/audio_agent.py --mode file --file_path /path/to/audio.wav
```

3) 运行备忘助手（本地模型）  [当前未使用]

```bash
python src/agents/memo_assistant.py --text-file src/inputs/对话记录.txt --model-path /path/to/local/model
```

注意：`memo_assistant` 默认使用大模型路径 `DEFAULT_MODEL_PATH`，请根据本机环境调整（需大量显存或使用 CPU）。

## 配置与提示模板
- 提示模板位于 `src/prompts/`，包括 `memo_assistant_system.txt`、`memo_assistant_user.txt`、`api_alpha_prompt.txt`、`api_beta_prompt.txt`、`api_charlie_prompt.txt` 等。修改这些文件可以调整模型的行为与输出格式。

## 隐私与日志
- `memo_assistant` 内置隐私脱敏逻辑，会将原始文本与脱敏结果保存到 `privacy/` 目录。
- 转写结果由 `audio_agent.TranscriptionCollector` 保存为 `transcription_result.json`（或其他指定路径）。

## 开发与调试提示
- 如果缺少 `pyaudio` 或麦克风设备，`main.py` 会跳过音频节点。你可以手动运行 `audio_agent.py` 来调试音频输入。 
- 若使用本地模型，请确保 `transformers`、`torch` 版本兼容，并根据 GPU 可用性调整 `memo_assistant.py` 中的 `device_map` 与 `torch_dtype` 设置。
- 多模态 Agent 使用 base64 编码图片并调用兼容 Dashscope 的 API；如遇到 API 错误，请检查 `DASHSCOPE_API_KEY` 与网络连通性。

## 常见命令汇总
```bash
# 安装依赖
pip install -r src/requirements.txt

# 运行主流程
python src/main.py

# 单独运行音频（麦克风/文件）
python src/agents/audio_agent.py --mode mic 7 6 （可以指定输入设备索引，在测试中可以自由切换可用麦克风）
python src/agents/audio_agent.py --mode file --file_path ./sample.wav

# 运行本地备忘助手（示例）
python src/agents/memo_assistant.py --text-file src/inputs/对话记录.txt
```



---

*自动生成 README（基于仓库 `src/` 内容）。若需要更详细的使用示例、API Key 管理、或 CI/部署 指南，请告诉我你想补充的部分。*
