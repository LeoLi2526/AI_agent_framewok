import os
import sys
import json
from typing import TypedDict, Annotated, Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler


# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
load_dotenv(dotenv_path=os.path.join(current_dir, ".env"))
sys.path.append(project_root)



try:
    import src.agents.audio_agent as audio_runner
    import pyaudio
    print("Successfully imported audio_runner and pyaudio")
except ImportError as e:
    print(f"Warning: Could not import audio modules: {e}")
    audio_runner = None

try:
    from src.utils.privacy_utils import save_privacy_info, sanitize_privacy_info
    print("Successfully imported privacy utils")
except ImportError as e:
    print(f"Warning: Could not import privacy utils: {e}")
    def save_privacy_info(*args, **kwargs): pass
    def sanitize_privacy_info(text): return text


try:
    from src.agents.multimodal_agent import EventDescriptionAgent
    print("Successfully imported EventDescriptionAgent")
except ImportError as e:
    print(f"Warning: Could not import EventDescriptionAgent: {e}")
    EventDescriptionAgent = None



# =====================State Settings=====================

class AgentState(TypedDict):
    '''Inputs'''
    dialogue_content: str
    user_habits: str
    emergency_event: str
    reminder_examples: str
    
    '''Intermediate States'''
    memo_list: str        
    alpha_reminder: str   
    beta_reminder: str    
    
    '''Outputs'''
    final_output: str     

# ===================Models Initialization=======================

api_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="deepseek-v3", 
    temperature=0.0,
    model_kwargs={"seed": 42}
)

# DeepSeek API
deepseek_official_llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-reasoner", 
    temperature=0.0,
    model_kwargs={"seed": 42}
)

# ====================Nodes Defination======================

def node_audio_agent(state: AgentState):
    """
    Node 0: Audio Agent (Voice Recognition)
    """
    print("--- Node 0: Audio Agent ---")
    
    if audio_runner is None:
        print("Audio modules not available. Skipping Audio Agent.")
        return {}

    # Check for microphones and list all available input devices
    input_device_indices = []
    try:       
        p = pyaudio.PyAudio()
        
        # Get default input device index to prioritize it
        default_device_index = None
        try:
            default_info = p.get_default_input_device_info()
            default_device_index = default_info['index']
        except Exception:
            pass

        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                input_device_indices.append(i)
        
        # Move default device to the front of the list
        if default_device_index is not None and default_device_index in input_device_indices:
            input_device_indices.remove(default_device_index)
            input_device_indices.insert(0, default_device_index)
            
        p.terminate()
    except Exception as e:
        print(f"Audio device check failed: {e}. Skipping Audio Agent.")
        return {}
    
    if not input_device_indices:
        print("No input devices found. Skipping Audio Agent.")
        return {}
        
    try:
        # Initialize Dashscope API Key
        audio_runner.init_dashscope_api_key()
        
        collector = audio_runner.TranscriptionCollector()
        callback = audio_runner.MyCallback(collector)
        
        conversation = audio_runner.OmniRealtimeConversation(
            model='qwen3-asr-flash-realtime',
            url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime',
            callback=callback,
        )
        
        transcription_params = audio_runner.TranscriptionParams(
            language='zh',
            sample_rate=16000,
            input_audio_format="pcm",
            corpus_text="这是一段中文对话"
        )
        
        conversation.connect()
        conversation.update_session(
            output_modalities=[audio_runner.MultiModality.TEXT],
            enable_input_audio_transcription=True,
            transcription_params=transcription_params,
        )
        
        print(f"Starting Audio Agent with devices: {input_device_indices}")
        # Pass the list of detected device indices
        audio_runner.run_mic_mode(conversation, collector, device_indices=input_device_indices)
        
        audio_text = collector.full_text
        print(f"Captured Audio Text: {audio_text}")
        
        if audio_text:
            # Append audio text to dialogue content
            original_dialogue = state.get('dialogue_content', "")
            new_dialogue = original_dialogue + "[语音输入记录]:" + audio_text
            return {"dialogue_content": new_dialogue}
            
    except Exception as e:
        print(f"Error running Audio Agent: {e}")
        
    return {}

def node_local_memo(state: AgentState):
    """
    Node 1: API模型提取备忘录 (替代原本地模型)
    """
    print("--- Node 1: API Memo Agent ---")
    
    # Load prompts
    prompts_dir = os.path.join(current_dir, "prompts")
    with open(os.path.join(prompts_dir, "memo_assistant_system.txt"), "r", encoding="utf-8") as f:
        system_prompt = f.read()
    with open(os.path.join(prompts_dir, "memo_assistant_user.txt"), "r", encoding="utf-8") as f:
        user_prompt_template = f.read()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt_template)
    ])
    
    chain = prompt | deepseek_official_llm | StrOutputParser()
    
    # Invoke chain
    memo_result = chain.invoke({"text": state["dialogue_content"]})
    
    # Sanitize and save privacy info
    sanitized_memo = sanitize_privacy_info(memo_result)
    save_privacy_info(state["dialogue_content"], sanitize_privacy_info(state["dialogue_content"]), sanitized_memo)
    
    return {"memo_list": sanitized_memo}

def node_api_alpha(state: AgentState):
    """
    Node 2: API模型 Alpha (突发事件 vs 习惯/备忘)
    """
    print("--- Node 2: API Alpha Agent ---")
    
    emergency_file_path = os.path.join(current_dir, "inputs/processed_emergency.json")
    emergency_content = ""
    
    # Read emergency event from file
    if os.path.exists(emergency_file_path):
        try:
            with open(emergency_file_path, "r", encoding="utf-8") as f:
                emergency_content = f.read()
            print(f"Loaded emergency content from {emergency_file_path}")
        except Exception as e:
            print(f"Error reading emergency file: {e}")
    else:
         # Fallback to state if file doesn't exist (though it should)
         print(f"Warning: Emergency file not found at {emergency_file_path}. Using state content.")
         emergency_content = state.get('emergency_event', "")

    prompt_path = os.path.join(current_dir, "prompts/api_alpha_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template_str = f.read()
    
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    
    chain = prompt | api_llm | StrOutputParser()
    response = chain.invoke({
        "memo_list": state['memo_list'],
        "user_habits": state['user_habits'],
        "emergency_event": emergency_content,
        "reminder_examples": state['reminder_examples']
    })
    return {"alpha_reminder": response}

def node_api_beta(state: AgentState):
    """
    Node 3: API模型 Beta (冲突检测)
    """
    print("--- Node 3: API Beta Agent ---")
    
    prompt_path = os.path.join(current_dir, "prompts/api_beta_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template_str = f.read()
        
    prompt = ChatPromptTemplate.from_template(template_str)
    chain = prompt | api_llm | StrOutputParser()
    
    response = chain.invoke({
        "memo_list": state["memo_list"],
        "user_habits": state["user_habits"]
    })
    
    return {"beta_reminder": response}

def node_multimodal_agent(state: AgentState):
    """
    Node: Multimodal Agent (Emergency Event Analysis)
    """
    print("--- Node: Multimodal Agent ---")
    
    base_inputs_dir = os.path.join(current_dir, "inputs/Emergency_information")
    examples_file = os.path.join(current_dir, "inputs/examples.json") # Corrected path assuming examples.json is in inputs now or multimodel/examples.json needs to be moved
    # Actually examples.json was in multimodel/examples.json. I should check where I copied it.
    # I copied inputs/examples.json to src/inputs/examples.json.
    # But multimodel/examples.json might be different. Let's assume src/inputs/examples.json is the one for now or I need to copy the multimodel one too.
    # Let's check the original file structure again.
    # inputs/examples.json existed. multimodel/examples.json existed.
    # I should probably use the one from multimodel for the multimodal agent.
    # I will update this path to point to a specific location later if needed.
    # For now let's use src/inputs/examples.json if it exists, or src/agents/examples.json if I copy it there.
    # I will assume I need to copy multimodel/examples.json to src/agents/multimodal_examples.json to be safe.
    
    # Wait, I see I copied inputs folder to src/inputs.
    # I should also copy multimodel/examples.json to src/agents/multimodal_examples.json
    
    examples_file = os.path.join(current_dir, "agents/multimodal_examples.json")
    output_file_path = os.path.join(current_dir, "inputs/processed_emergency.json")
    
    if EventDescriptionAgent is None:
        print("EventDescriptionAgent not available.")
        return {}
        
    try:
        agent = EventDescriptionAgent(examples_path=examples_file)
    except Exception as e:
        print(f"Failed to init EventDescriptionAgent: {e}")
        return {}

    if os.path.exists(base_inputs_dir):
        subfolders = [f.path for f in os.scandir(base_inputs_dir) if f.is_dir()]
        subfolders.sort()
        
        if subfolders:
            # Use the latest folder
            latest_folder = subfolders[-1]
            print(f"Processing latest event folder: {latest_folder}")
            result = agent.analyze_event(latest_folder)
            
            # Convert result to string and save to file
            if isinstance(result, (dict, list)):
                result_str = json.dumps(result, ensure_ascii=False, indent=2)
            else:
                result_str = str(result)
            
            # Save to JSON file
            try:
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(result_str)
                print(f"Emergency event saved to: {output_file_path}")
            except Exception as e:
                print(f"Error saving emergency event to file: {e}")

            print("Multimodal Agent Output:", result_str)
            return {"emergency_event": result_str}
        else:
            print("No event folders found.")
    else:
        print(f"Input directory not found: {base_inputs_dir}")
    
    return {}

def node_api_charlie(state: AgentState):
    """
    Node 4: API模型 Charlie (整合输出)
    """
    print("--- Node 4: API Charlie Agent ---")
    
    prompt_path = os.path.join(current_dir, "prompts/api_charlie_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template_str = f.read()
        
    prompt = ChatPromptTemplate.from_template(template_str)
    chain = prompt | deepseek_official_llm | StrOutputParser()
    
    response = chain.invoke({
        "memo_list": state["memo_list"],
        "alpha_reminder": state["alpha_reminder"],
        "beta_reminder": state["beta_reminder"]
    })
    
    return {"final_output": response}

# ====================Graph Construction======================

workflow = StateGraph(AgentState)

workflow.add_node("multimodal_agent", node_multimodal_agent)
workflow.add_node("audio_agent", node_audio_agent)
workflow.add_node("local_memo", node_local_memo)
workflow.add_node("api_alpha", node_api_alpha)
workflow.add_node("api_beta", node_api_beta)
workflow.add_node("api_charlie", node_api_charlie)

# Edges
# Start -> Multimodal Agent
workflow.set_entry_point("multimodal_agent")

# Multimodal Agent -> Audio Agent
workflow.add_edge("multimodal_agent", "audio_agent")

# Audio Agent -> Local Memo
workflow.add_edge("audio_agent", "local_memo")

# Local Memo -> Alpha & Beta 
workflow.add_edge("local_memo", "api_alpha")
workflow.add_edge("local_memo", "api_beta")

# Alpha & Beta -> Charlie
workflow.add_edge("api_alpha", "api_charlie")
workflow.add_edge("api_beta", "api_charlie")

# Charlie -> End
workflow.add_edge("api_charlie", END)

app = workflow.compile()


# =============== Main Execution =========================

if __name__ == "__main__":
    INPUT_DIR = os.path.join(current_dir, "inputs")
    DIALOGUE_FILE = os.path.join(INPUT_DIR, "对话记录.txt")
    HABITS_FILE = os.path.join(INPUT_DIR, "行为习惯.txt")
    EXAMPLES_FILE = os.path.join(INPUT_DIR, "examples.json")

    def read_file_content(file_path):
        """读取文件内容，如果文件不存在则返回提示信息"""
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

    print("Reading input files from:", INPUT_DIR)
    
    dialogue = read_file_content(DIALOGUE_FILE)
    habits = read_file_content(HABITS_FILE)
    examples = read_file_content(EXAMPLES_FILE)

    print("Starting LangGraph Workflow...")
    
    inputs = {
        "dialogue_content": dialogue,
        "user_habits": habits,
        "reminder_examples": examples
    }
    
    # 运行图
    # 初始化 Langfuse Handler
    # 依赖环境变量：LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST/LANGFUSE_BASE_URL
    # 确保兼容性
    if os.getenv("LANGFUSE_BASE_URL") and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL")
        
    langfuse_handler = CallbackHandler()
    
    print("Langfuse monitoring enabled.")
    
    result = app.invoke(inputs, config={"callbacks": [langfuse_handler]})
    
    print("================ FINAL OUTPUT ================")
    print(result["final_output"])
    print("==============================================")
