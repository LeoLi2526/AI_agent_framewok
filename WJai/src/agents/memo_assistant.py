from __future__ import annotations

import argparse
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


DEFAULT_MODEL_PATH = "/home/leoli/Downloads_for_ai/Models/Qwen3-4B-Instruct-2507"

_model_cache = {}
_tokenizer_cache = {}

class MemoAssistant:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        global _model_cache, _tokenizer_cache

        if model_path in _tokenizer_cache:
            self.tokenizer = _tokenizer_cache[model_path]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            _tokenizer_cache[model_path] = self.tokenizer

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                device_map = "auto"  
                print(f"检测到 {num_gpus} 个GPU，启用多GPU并行处理")                
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
            else:
                device_map = "auto"
        else:
            device_map = {"": "cpu"}
        if model_path in _model_cache:
            self.model = _model_cache[model_path]
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  
                use_safetensors=True,    
            )
            _model_cache[model_path] = self.model
        self.model.eval()

        model_gen_config = self.model.generation_config
        if model_gen_config is None:
            model_gen_config = GenerationConfig.from_model_config(self.model.config)

        default_gen = deepcopy(model_gen_config)
        default_gen.do_sample = False  
        default_gen.temperature = 0.1 
        default_gen.top_p = 0.7        
        default_gen.repetition_penalty = 1.1  
        default_gen.max_new_tokens = 512      
        default_gen.num_beams = 1             

        if generation_kwargs:
            for key, value in generation_kwargs.items():
                setattr(default_gen, key, value)

        if default_gen.pad_token_id is None:
            default_gen.pad_token_id = self.tokenizer.pad_token_id
        if default_gen.eos_token_id is None and self.tokenizer.eos_token_id is not None:
            default_gen.eos_token_id = self.tokenizer.eos_token_id
        if default_gen.bos_token_id is None and self.tokenizer.bos_token_id is not None:
            default_gen.bos_token_id = self.tokenizer.bos_token_id

        self.generation_config = default_gen

    def sanitize_privacy_info(self, text: str) -> str:
        """对隐私信息进行脱敏处理"""
        text = re.sub(r'(1[3-9]\d)(\d{4})(\d{4})', r'\1****\3', text)
        text = re.sub(r'(\d{6})(\d{8})(\d{4})', r'\1********\3', text)
        text = re.sub(r'(\d{4})(\d{8,11})(\d{4})', r'\1********\3', text)
        text = re.sub(r'\b([A-Za-z0-9._%+-]{2})([A-Za-z0-9._%+-]*)(@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b', r'\1****\3', text)
        address_patterns = [
            r'([\u4e00-\u9fff]+[省市县区][\u4e00-\u9fff]+[街道路巷][\u4e00-\u9fff]*\d+号?)([\u4e00-\u9fff\w\d\s]*)',
            r'([\u4e00-\u9fff]+[省市县区][\u4e00-\u9fff]+[小区社区][\u4e00-\u9fff]*\d+栋?)([\u4e00-\u9fff\w\d\s]*)',
            r'([\u4e00-\u9fff]+[大厦楼座][\u4e00-\u9fff]*\d+层?)([\u4e00-\u9fff\w\d\s]*)',
        ]
        for pattern in address_patterns:
            text = re.sub(pattern, r'\1****', text)
        text = re.sub(r'\b([A-Za-z])(\d+)(\d{3})\b', r'\1****\3', text)
        amount_pattern = r'(\d+\.?\d*)\s*(元|美元|镑|欧元|￥|\$)?'
        def amount_repl(match):
            digits = re.sub(r'[^\d]', '', match.group(1))  
            if len(digits) < 3:
                return match.group(0)  
            masked_digits = digits[0] + '*' * (len(digits) - 2) + digits[-1]
            unit = match.group(2) if match.group(2) else ''
            return masked_digits + ' ' + unit
        text = re.sub(amount_pattern, amount_repl, text)
        return text

    def save_privacy_info(self, original_text: str, sanitized_text: str, memo_output: str = "") -> None:
        """将原始隐私信息和处理结果保存到隐私文件夹"""
        privacy_dir = "privacy"
        if not os.path.exists(privacy_dir):
            os.makedirs(privacy_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{privacy_dir}/privacy_info_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=== 原始文本 ===\n")
            f.write(original_text)
            f.write("\n\n=== 脱敏后文本 ===\n")
            f.write(sanitized_text)
            if memo_output:
                f.write("\n\n=== 模型生成的备忘录 ===\n")
                f.write(memo_output)
            f.write("\n\n=== 处理时间 ===\n")
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def build_messages(self, text: str) -> List[Dict[str, str]]:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompts_dir = os.path.join(base_dir, "prompts")

        with open(os.path.join(prompts_dir, "memo_assistant_system.txt"), "r", encoding="utf-8") as f:
            system_prompt = f.read()
        with open(os.path.join(prompts_dir, "memo_assistant_user.txt"), "r", encoding="utf-8") as f:
            user_prompt_template = f.read()
        
        # 格式化用户提示
        user_prompt = user_prompt_template.format(text=text.strip())

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @torch.no_grad()
    def generate_memo(self, text: str) -> str:
        messages = self.build_messages(text)
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        input_ids = input_ids.to(self.model.device)
        attention_mask = torch.ones_like(input_ids, device=self.model.device)

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
        )

        generated_ids = output_ids[:, input_ids.shape[-1]:]
        raw_output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        sanitized_output = self.sanitize_privacy_info(raw_output)
        
        self.save_privacy_info(text, self.sanitize_privacy_info(text), sanitized_output)
        
        return sanitized_output




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen 车载备忘助手 CLI")
    parser.add_argument(
        "--text-file",
        type=str,
        required=False,
        help="包含原始文本的文件路径。如果未提供，使用参数文件中的默认值。",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        help="Qwen 模型所在路径。如果未提供，使用参数文件中的默认值。",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="运行时忽略任何 HTTP/HTTPS 代理设置。",
    )
    return parser.parse_args()


def read_text_file(text_file: str) -> str:
    """从文件读取文本内容"""
    with open(text_file, "r", encoding="utf-8") as f:
        return f.read()


def configure_proxy(no_proxy: bool = False) -> None:
    if no_proxy:
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        return

    proxy_url = "http://10.157.197.169:7890"
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ.setdefault(key, proxy_url)


def save_processed_results(text: str, memo_output: str) -> None:
    """将处理结果按照格式保存到文件"""
    results_dir = "processed_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results_dir}/result_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== 处理结果报告 ===\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write("=== 生成的备忘录 ===\n")
        f.write(memo_output)
        f.write("\n\n")
        f.write("=== 统计信息 ===\n")
        memo_lines = [line for line in memo_output.split('\n') if line.strip() and line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.'))]
        f.write(f"生成的备忘录条目: {len(memo_lines)}\n")
        f.write(f"原始文本长度: {len(text)} 字符\n")
        f.write(f"备忘录长度: {len(memo_output)} 字符\n")
    
    print(f"处理结果已保存到: {filename}")


def load_model_params() -> dict:
    """从参数文件加载模型参数和路径配置"""
    params_file = "model_params.json"
    if not os.path.exists(params_file):
        print(f"参数文件 {params_file} 不存在，使用默认参数")
        return {
            "max_new_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "default_input_file": "input.txt",
            "default_output_dir": "processed_results",
            "prompt_dir": "prompts",
            "model_path": "/home/leoli/Downloads_for_ai/Models/Qwen3-4B-Instruct-2507"
        }
    
    with open(params_file, "r", encoding="utf-8") as f:
        params_data = json.load(f)

    memo_params = params_data.get("memo_assistant", {})
    path_params = params_data.get("paths", {})
    memo_params.setdefault("default_input_file", "input.txt")
    memo_params.setdefault("default_output_dir", "processed_results")
    memo_params.setdefault("prompt_dir", "prompts")
    path_params.setdefault("model_path", "/home/leoli/Downloads_for_ai/Models/Qwen3-4B-Instruct-2507")
    memo_params.update(path_params)
    return memo_params


def parse_memo_events(memo_output: str) -> List[Dict[str, str]]:
    """解析备忘录输出，提取各个事件信息"""
    events = []
    lines = memo_output.strip().split('\n')
    
    current_event = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 检测事件编号（如 "1.", "2.", 等）
        if re.match(r'^\d+\.', line):
            if current_event:  # 保存前一个事件
                events.append(current_event)
            current_event = {'raw_text': line, 'number': len(events) + 1}
        elif current_event:
            current_event['raw_text'] += '\n' + line
    
    # 添加最后一个事件
    if current_event:
        events.append(current_event)
    
    return events


def save_processed_results_split(text: str, memo_output: str) -> None:
    """将备忘录结果按事件分割保存到不同文件"""
    events = parse_memo_events(memo_output)
    
    if not events:
        print("警告: 未检测到事件，使用标准保存方式")
        save_processed_results(text, memo_output)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    events_dir = f"processed_results/events_{timestamp}"
    os.makedirs(events_dir, exist_ok=True)
    
    summary_file = f"{events_dir}/summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=== 备忘录事件分割报告 ===\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"检测到事件数量: {len(events)}\n")
        f.write(f"原始文本长度: {len(text)} 字符\n")
        f.write(f"备忘录总长度: {len(memo_output)} 字符\n\n")
        f.write("=== 事件概览 ===\n")
        
        for i, event in enumerate(events, 1):
            f.write(f"事件 {i}: {event['raw_text'][:100]}...\n")
    
    for i, event in enumerate(events, 1):
        event_file = f"{events_dir}/event_{i:03d}.txt"
        with open(event_file, "w", encoding="utf-8") as f:
            f.write("=== 备忘录事件详情 ===\n")
            f.write(f"事件编号: {i}\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"事件内容长度: {len(event['raw_text'])} 字符\n\n")
            f.write("=== 事件内容 ===\n")
            f.write(event['raw_text'])
    
    save_processed_results(text, memo_output)
    
    print(f"事件分割结果已保存到: {events_dir}")
    print(f"检测到 {len(events)} 个事件，已分别保存到事件文件")


def run_memo_assistant(text_file: str = None, model_path: str = None, no_proxy: bool = False) -> str:
    """运行备忘录助手的主要逻辑"""
    configure_proxy(no_proxy)

    params = load_model_params()
    print(f"加载参数配置: {params}")
    
    if text_file is None:
        text_file = params.get("default_input_file", "input.txt")
    
    if model_path is None:
        model_path = params.get("model_path", "/home/leoli/Downloads_for_ai/Models/Qwen3-4B-Instruct-2507")
    
    if not os.path.exists(text_file):
        print(f"错误: 输入文件 {text_file} 不存在")
        return ""

    generation_kwargs = {k: v for k, v in params.items() if k in ["max_new_tokens", "temperature", "top_p", "repetition_penalty", "do_sample"]}
    print(f"使用生成参数: {generation_kwargs}")

    assistant = MemoAssistant(
        model_path=model_path,
        generation_kwargs=generation_kwargs,
    )

    text = read_text_file(text_file)
    print(f"读取输入文件: {text_file}")
    print(f"文本长度: {len(text)} 字符")
    
    memo_output = assistant.generate_memo(text)

    print("=== 备忘录输出 ===")
    print(memo_output)
    
    save_processed_results_split(text, memo_output)
    
    return memo_output


def main() -> None:
    """命令行入口点"""
    args = parse_args()
    run_memo_assistant(
        text_file=args.text_file,
        model_path=args.model_path,
        no_proxy=args.no_proxy
    )


if __name__ == "__main__":
    main()

