import json
import re
import ast
from typing import Union, Optional

def ultimate_json_score_extractor(data: Union[str, dict, None], 
                                 score_key: str = 'correctness_score') -> Optional[float]:
    """
    最完整版的JSON分数提取函数
    
    Args:
        data: 输入数据，可以是字符串、字典或None
        score_key: 要提取的键名，默认为'correctness_score'
    
    Returns:
        float: 提取的分数值，失败返回None
    """
    
    # === 第1层：基础验证 ===
    if data is None:
        return None
    
    # 如果已经是字典，直接提取
    if isinstance(data, dict):
        value = data.get(score_key)
        return float(value) if value is not None else None
    
    # 如果不是字符串，尝试转换
    if not isinstance(data, str):
        try:
            data = str(data)
        except:
            return None
    
    # === 第2层：字符串预处理 ===
    # 移除BOM和控制字符
    data = data.strip()
    if data.startswith('\ufeff'):  # UTF-8 BOM
        data = data[1:]
    if data.startswith('\ufffe') or data.startswith('\xef\xbb\xbf'):  # 其他BOM
        data = data[3:]
    
    # 移除多余的空白字符
    data = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', data)
    
    # === 第3层：标准JSON解析 ===
    try:
        parsed = json.loads(data)
        if isinstance(parsed, dict) and score_key in parsed:
            value = parsed[score_key]
            return float(value) if value is not None else None
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # === 第3.5层：处理JSON后有额外内容的情况 ===
    # 尝试找到第一个完整的JSON对象
    if '{' in data and '}' in data:
        try:
            # 找到第一个{的位置
            start = data.find('{')
            # 从该位置开始，找到匹配的}
            brace_count = 0
            end = start
            for i, char in enumerate(data[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            if end > start:
                json_part = data[start:end]
                parsed = json.loads(json_part)
                if isinstance(parsed, dict) and score_key in parsed:
                    value = parsed[score_key]
                    return float(value) if value is not None else None
        except:
            pass
    
    # === 第4层：修复常见JSON问题 ===
    fixes = [
        # 修复单引号
        lambda x: x.replace("'", '"'),
        # 修复转义字符
        lambda x: x.replace('\\', '\\\\'),
        # 修复未转义的换行符
        lambda x: x.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t'),
        # 修复尾部逗号
        lambda x: re.sub(r',\s*}', '}', x),
        lambda x: re.sub(r',\s*]', ']', x),
        # 修复缺少引号的键
        lambda x: re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\\1"\\2":', x),
    ]
    
    for fix in fixes:
        try:
            fixed_data = fix(data)
            parsed = json.loads(fixed_data)
            if isinstance(parsed, dict) and score_key in parsed:
                value = parsed[score_key]
                return float(value) if value is not None else None
        except:
            continue
    
    # === 第5层：AST解析（处理Python字典格式）===
    try:
        # 尝试将类似Python字典的字符串转换
        parsed = ast.literal_eval(data)
        if isinstance(parsed, dict) and score_key in parsed:
            value = parsed[score_key]
            return float(value) if value is not None else None
    except:
        pass
    
    # === 第6层：正则表达式直接提取 ===
    patterns = [
        # 标准格式: "correctness_score": 0.5
        rf'"{score_key}":\s*([0-9]*\.?[0-9]+)',
        # 单引号格式: 'correctness_score': 0.5
        rf"'{score_key}':\s*([0-9]*\.?[0-9]+)",
        # 无引号格式: correctness_score: 0.5
        rf'{score_key}:\s*([0-9]*\.?[0-9]+)',
        # 带空格格式: "correctness_score" : 0.5
        rf'"{score_key}"\s*:\s*([0-9]*\.?[0-9]+)',
    ]
    
    for pattern in patterns:
        try:
            matches = re.findall(pattern, data, re.IGNORECASE)
            if matches:
                return float(matches[0])
        except:
            continue
    
    # === 第7层：模糊匹配 ===
    # 查找任何包含score_key的行
    lines = data.split('\n')
    for line in lines:
        if score_key.lower() in line.lower():
            # 提取该行中的数字
            numbers = re.findall(r'([0-9]*\.?[0-9]+)', line)
            if numbers:
                try:
                    return float(numbers[0])
                except:
                    continue
    
    # === 第8层：最后尝试 - 查找任何浮点数 ===
    # 如果JSON看起来只包含一个分数值
    if len(data.strip()) < 100:  # 短字符串
        numbers = re.findall(r'\b([0-1]\.?[0-9]*)\b', data)
        if len(numbers) == 1:
            try:
                return float(numbers[0])
            except:
                pass
    
    # === 完全失败 ===
    return None