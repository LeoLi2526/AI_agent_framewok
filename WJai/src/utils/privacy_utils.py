import re
import os
from datetime import datetime

def sanitize_privacy_info(text: str) -> str:
    """对隐私信息进行脱敏处理"""
    if not text:
        return ""
    text = re.sub(r'(1[3-9]\d)(\d{4})(\d{4})', r'\1****\3', text)
    text = re.sub(r'(\d{6})(\d{8})(\d{4})', r'\1********\3', text)
    text = re.sub(r'(\d{4})(\d{8,11})(\d{4})', r'\1********\3', text)
    text = re.sub(r'([A-Za-z0-9._%+-]{2})([A-Za-z0-9._%+-]*)(@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', r'\1****\3', text)
    address_patterns = [
        r'([\u4e00-\u9fff]+[省市县区][\u4e00-\u9fff]+[街道路巷][\u4e00-\u9fff]*\d+号?)([\u4e00-\u9fff\w\d\s]*)',
        r'([\u4e00-\u9fff]+[省市县区][\u4e00-\u9fff]+[小区社区][\u4e00-\u9fff]*\d+栋?)([\u4e00-\u9fff\w\d\s]*)',
        r'([\u4e00-\u9fff]+[大厦楼座][\u4e00-\u9fff]*\d+层?)([\u4e00-\u9fff\w\d\s]*)',
    ]
    for pattern in address_patterns:
        text = re.sub(pattern, r'\1****', text)
    text = re.sub(r'([A-Za-z])(\d+)(\d{3})', r'\1****\3', text)
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

def save_privacy_info(original_text: str, sanitized_text: str, memo_output: str = "") -> None:
    """将原始隐私信息和处理结果保存到隐私文件夹"""
    # Assuming privacy directory is at the project root level, similar to how it was used.
    # We should use absolute path if possible or relative to CWD.
    # In main.py, project_root is defined. But here we don't have it.
    # We will assume CWD is project root or use a relative path "privacy"
    
    privacy_dir = "privacy"
    if not os.path.exists(privacy_dir):
        # Try to find privacy dir relative to this file? 
        # But for now let's assume running from project root.
        try:
            os.makedirs(privacy_dir)
        except OSError:
            # Fallback if we can't create in CWD, maybe inside src?
            # But let's stick to "privacy" as per original code.
            pass
            
    if not os.path.exists(privacy_dir):
        print(f"Warning: Could not access privacy directory '{privacy_dir}'")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(privacy_dir, f"privacy_info_{timestamp}.txt")
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=== 原始文本 ===")
            f.write(original_text)
            f.write("=== 脱敏后文本 ===")
            f.write(sanitized_text)
            if memo_output:
                f.write("=== 模型生成的备忘录 ===")
                f.write(memo_output)
            f.write("=== 处理时间 ===")
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        print(f"Error saving privacy info: {e}")
