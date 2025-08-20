import re
import json


def safe_json_loads(s):
    # 使用正则匹配 ```json 或 ``` 包裹的内容
    pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(pattern, s, re.DOTALL)

    if match:
        json_str = match.group(1)
    else:
        # 没有代码块格式，尝试直接找大括号包围的 JSON
        brace_pattern = r"(\{.*\})"
        match = re.search(brace_pattern, s, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            raise ValueError("未找到可解析的 JSON 字符串。")
    json_str = json_str.strip()

    # 去除非法控制字符（如未转义的换行符）
    cleaned = re.sub(r'(?<!\\)[\n\r\t]', ' ', json_str)

    return json.loads(cleaned)