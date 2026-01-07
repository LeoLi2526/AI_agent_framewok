from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


DEFAULT_MODEL_PATH = "/home/leoli/Downloads_for_ai/Models/Qwen3-4B-Instruct-2507"


@dataclass
class Habit:
    """描述用于匹配习惯行为的规则。"""

    name: str
    all_keywords: List[str] = field(default_factory=list)
    any_keywords: List[str] = field(default_factory=list)
    description: Optional[str] = None

    def matches(self, text: str) -> bool:
        """检查文本是否匹配该习惯"""
        lowered = text.lower()
        if self.all_keywords and not any(keyword.lower() in lowered for keyword in self.all_keywords):
            return False
        if self.any_keywords and not any(keyword.lower() in lowered for keyword in self.any_keywords):
            return False
        return bool(self.all_keywords or self.any_keywords)


def load_habits(habit_path: str) -> List[Habit]:
    """从 JSON 文件加载习惯配置。"""
    with open(habit_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("习惯配置文件必须是由对象组成的列表。")

    habits: List[Habit] = []
    for item in data:
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError("每条习惯配置必须至少包含 name 字段。")
        habits.append(
            Habit(
                name=item["name"],
                all_keywords=item.get("all_keywords", []),
                any_keywords=item.get("any_keywords", []),
                description=item.get("description"),
            )
        )
    return habits


class HabitDetector:
    """封装 Qwen 模型调用与智能习惯检测逻辑。"""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 多GPU配置
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.eval()

        model_gen_config = self.model.generation_config
        if model_gen_config is None:
            model_gen_config = GenerationConfig.from_model_config(self.model.config)

        default_gen = model_gen_config
        default_gen.do_sample = False
        default_gen.temperature = 0.1
        default_gen.top_p = 0.7
        default_gen.max_new_tokens = 256
        default_gen.num_beams = 1

        if generation_kwargs:
            for key, value in generation_kwargs.items():
                setattr(default_gen, key, value)

        self.generation_config = default_gen

    def build_messages(self, memo_text: str, habits: List[Habit]) -> List[Dict[str, str]]:
        """构建模型输入消息"""
        habit_descriptions = "\n".join([
            f"{i+1}. {habit.name}: {habit.description or '无描述'} "
            f"(关键词: {', '.join(habit.all_keywords + habit.any_keywords)})"
            for i, habit in enumerate(habits)
        ])

        with open("prompts/habit_detector_system.txt", "r", encoding="utf-8") as f:
            system_prompt_template = f.read()
        
        with open("prompts/habit_detector_user.txt", "r", encoding="utf-8") as f:
            user_prompt_template = f.read()
        
        system_prompt = system_prompt_template.format(habit_descriptions=habit_descriptions)
        user_prompt = user_prompt_template.format(memo_text=memo_text)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @torch.no_grad()
    def detect_habits_intelligently(self, memo_text: str, habits: List[Habit]) -> str:
        """使用模型智能检测备忘录中的习惯"""
        messages = self.build_messages(memo_text, habits)
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
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()




def read_memo_file(memo_file: str) -> str:
    """读取备忘录文件内容"""
    with open(memo_file, "r", encoding="utf-8") as f:
        return f.read()


def configure_proxy(no_proxy: bool = False) -> None:
    """配置网络代理设置"""
    if no_proxy:
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
    else:
        os.environ['http_proxy'] = 'http://10.157.197.169:7890'
        os.environ['https_proxy'] = 'http://10.157.197.169:7890'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="智能习惯检测助手")
    parser.add_argument(
        "--memo-file",
        type=str,
        required=False,
        help="包含备忘录的文件路径。如果未提供，将自动查找最新结果文件。",
    )
    parser.add_argument(
        "--memo-dir",
        type=str,
        required=False,
        help="包含多个备忘录文件的目录。如果提供，将批量处理该目录下的所有备忘录文件。",
    )
    parser.add_argument(
        "--events-dir",
        type=str,
        required=False,
        help="包含事件文件的目录（由备忘录助手生成）。如果提供，将处理该目录下的所有事件文件。",
    )
    parser.add_argument(
        "--habits",
        type=str,
        required=False,
        help="习惯配置 JSON 文件路径。如果未提供，使用参数文件中的默认值。",
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
        help="不使用网络代理。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="批处理大小，每次处理的文件数量。默认：10",
    )

    return parser.parse_args()


def load_model_params() -> dict:
    """从参数文件加载模型参数和路径配置"""
    params_file = "model_params.json"
    if not os.path.exists(params_file):
        print(f"参数文件 {params_file} 不存在，使用默认参数")
        return {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "do_sample": False,
            "default_habits_file": "habits.json",
            "default_prompt_dir": "prompts",
            "model_path": "/home/leoli/Downloads_for_ai/Models/Qwen3-4B-Instruct-2507"
        }
    
    with open(params_file, "r", encoding="utf-8") as f:
        params_data = json.load(f)
    habit_params = params_data.get("habit_detector", {})
    path_params = params_data.get("paths", {})
    
    habit_params.setdefault("default_habits_file", "habits.json")
    habit_params.setdefault("default_prompt_dir", "prompts")
    path_params.setdefault("model_path", "/home/leoli/Downloads_for_ai/Models/Qwen3-4B-Instruct-2507")
    
    habit_params.update(path_params)
    return habit_params


def save_event_habit_results(events_dir: str, results: List[Tuple[str, str]]) -> None:
    """保存事件级别的习惯检测结果"""
    habits_dir = events_dir.replace("events_", "habits_")
    os.makedirs(habits_dir, exist_ok=True)
    
    summary_file = f"{habits_dir}/summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=== 事件习惯检测汇总报告 ===\n")
        f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"对应事件目录: {events_dir}\n")
        f.write(f"处理事件数量: {len(results)}\n\n")
        
        total_habits = 0
        for memo_file, habit_result in results:
            if not habit_result.startswith("处理错误"):
                habit_lines = [line for line in habit_result.split('\n') if line.strip() and line.startswith('-')]
                total_habits += len(habit_lines)
        
        f.write(f"检测到的习惯总数: {total_habits}\n\n")
        f.write("=== 事件详情 ===\n")
        
        for i, (memo_file, habit_result) in enumerate(results, 1):
            filename = os.path.basename(memo_file)
            if habit_result.startswith("处理错误"):
                f.write(f"事件 {i} ({filename}): 处理错误\n")
            else:
                habit_lines = [line for line in habit_result.split('\n') if line.strip() and line.startswith('-')]
                f.write(f"事件 {i} ({filename}): 检测到 {len(habit_lines)} 个习惯\n")
    
    for memo_file, habit_result in results:
        if not habit_result.startswith("处理错误"):
            filename = os.path.basename(memo_file)
            habit_filename = f"{habits_dir}/habit_{filename}"
            
            with open(habit_filename, "w", encoding="utf-8") as f:
                f.write("=== 事件习惯检测结果 ===\n")
                f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"对应事件文件: {memo_file}\n\n")
                f.write("=== 智能习惯检测结果 ===\n")
                f.write(habit_result)
    
    print(f"事件习惯检测结果已保存到: {habits_dir}")


def save_habit_detection_results(memo_file: str, habit_result: str, memo_text: str) -> str:
    """保存习惯检测结果到文件，与备忘录文件相对应"""
    results_dir = "processed_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    memo_filename = os.path.basename(memo_file)
    if memo_filename.startswith("result_") and memo_filename.endswith(".txt"):
        timestamp = memo_filename[7:-4]  # 提取 result_20251110_120000 中的时间戳部分
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    habit_filename = f"{results_dir}/habit_result_{timestamp}.txt"
    
    with open(habit_filename, "w", encoding="utf-8") as f:
        f.write("=== 习惯检测结果报告 ===\n")
        f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"对应备忘录文件: {memo_file}\n")
        f.write(f"备忘录长度: {len(memo_text)} 字符\n")
        f.write("\n")
        
        f.write("=== 智能习惯检测结果 ===\n")
        f.write(habit_result)
        f.write("\n\n")
        

    
    print(f"习惯检测结果已保存到: {habit_filename}")
    return habit_filename


def batch_detect_habits(memo_files: List[str], habits_file: str, model_path: str, generation_kwargs: dict, batch_size: int = 10) -> List[Tuple[str, str]]:
    """批量处理多个备忘录文件"""
    print(f"开始批量处理 {len(memo_files)} 个备忘录文件，批处理大小: {batch_size}")
    
    habits = load_habits(habits_file)
    print(f"加载了 {len(habits)} 个习惯配置")
    
    detector = HabitDetector(model_path=model_path, generation_kwargs=generation_kwargs)
    
    results = []
    batch_count = 0

    for i in range(0, len(memo_files), batch_size):
        batch_files = memo_files[i:i + batch_size]
        batch_count += 1
        
        print(f"\n=== 处理批次 {batch_count} ({len(batch_files)} 个文件) ===")
        
        for memo_file in batch_files:
            try:
                memo_text = read_memo_file(memo_file)
                print(f"处理文件: {os.path.basename(memo_file)} (长度: {len(memo_text)} 字符)")
                
                intelligent_result = detector.detect_habits_intelligently(memo_text, habits)
                results.append((memo_file, intelligent_result))

                habit_result_file = save_habit_detection_results(memo_file, intelligent_result, memo_text)
                
                habit_lines = [line for line in intelligent_result.split('\n') if line.strip() and line.startswith('-')]
                print(f"检测到 {len(habit_lines)} 个习惯")
                
            except Exception as e:
                print(f"处理文件 {memo_file} 时出错: {e}")
                results.append((memo_file, f"处理错误: {e}"))
    
    print(f"\n=== 批量处理完成 ===")
    print(f"成功处理: {len([r for r in results if not r[1].startswith('处理错误')])}/{len(memo_files)} 个文件")
    
    return results


def run_habit_detection(memo_file: str = None, memo_dir: str = None, events_dir: str = None, 
                       habits_file: str = None, model_path: str = None, no_proxy: bool = False, batch_size: int = 10) -> str:
    """运行习惯检测的主要逻辑"""
    configure_proxy(no_proxy)

    params = load_model_params()
    print(f"加载参数配置: {params}")
    
    if habits_file is None:
        habits_file = params.get("default_habits_file", "habits.json")
    
    if model_path is None:
        model_path = params.get("model_path", "/home/leoli/Downloads_for_ai/Models/Qwen3-4B-Instruct-2507")
    
    generation_kwargs = {k: v for k, v in params.items() if k in ["max_new_tokens", "temperature", "top_p", "repetition_penalty", "do_sample"]}
    print(f"使用生成参数: {generation_kwargs}")
    
    memo_files = []
    
    if events_dir:
        if not os.path.exists(events_dir):
            print(f"错误: 事件目录 {events_dir} 不存在")
            return ""
        
        for filename in os.listdir(events_dir):
            if filename.startswith("event_") and filename.endswith(".txt"):
                memo_files.append(os.path.join(events_dir, filename))
        
        if not memo_files:
            print(f"错误: 事件目录 {events_dir} 中未找到事件文件")
            return ""
        
        memo_files.sort()  
        print(f"在事件目录 {events_dir} 中找到 {len(memo_files)} 个事件文件")
        results = batch_detect_habits(memo_files, habits_file, model_path, generation_kwargs, batch_size)
        
        save_event_habit_results(events_dir, results)
        
        return results[-1][1] if results else ""
    
    elif memo_dir:
        if not os.path.exists(memo_dir):
            print(f"错误: 目录 {memo_dir} 不存在")
            return ""
        
        for filename in os.listdir(memo_dir):
            if filename.startswith("result_") and filename.endswith(".txt"):
                memo_files.append(os.path.join(memo_dir, filename))
        
        if not memo_files:
            print(f"错误: 目录 {memo_dir} 中未找到备忘录文件")
            return ""
        
        memo_files.sort()  # 按文件名排序
        print(f"在目录 {memo_dir} 中找到 {len(memo_files)} 个备忘录文件")
        
        results = batch_detect_habits(memo_files, habits_file, model_path, generation_kwargs, batch_size)
        
        return results[-1][1] if results else ""
    
    else:
        if memo_file is None:
            results_dir = "processed_results"
            if os.path.exists(results_dir):
                result_files = [f for f in os.listdir(results_dir) if f.startswith("result_") and f.endswith(".txt")]
                if result_files:
                    result_files.sort(reverse=True)
                    memo_file = os.path.join(results_dir, result_files[0])
                    print(f"自动选择最新备忘录文件: {memo_file}")
                else:
                    print("错误: 未找到备忘录文件，请使用 --memo-file 或 --memo-dir 参数指定")
                    return ""
            else:
                print("错误: 未找到备忘录文件，请使用 --memo-file 或 --memo-dir 参数指定")
                return ""

        memo_files = [memo_file]
        
        habits = load_habits(habits_file)
        print(f"加载了 {len(habits)} 个习惯配置")
        
        memo_text = read_memo_file(memo_file)
        print(f"读取备忘录文件: {memo_file}")
        print(f"备忘录长度: {len(memo_text)} 字符")

        detector = HabitDetector(model_path=model_path, generation_kwargs=generation_kwargs)
        
        print("\n=== 智能习惯检测结果 ===")

        intelligent_result = detector.detect_habits_intelligently(memo_text, habits)
        print(intelligent_result)

        habit_result_file = save_habit_detection_results(memo_file, intelligent_result, memo_text)
        
        return intelligent_result


def main() -> None:
    """命令行入口点"""
    args = parse_args()
    run_habit_detection(
        memo_file=args.memo_file,
        memo_dir=args.memo_dir,
        events_dir=args.events_dir,
        habits_file=args.habits,
        model_path=args.model_path,
        no_proxy=args.no_proxy,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()