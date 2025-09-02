import time
import uuid

def generate_doi():
    # 获取当前时间戳（毫秒级）
    timestamp = int(time.time() * 1000)
    
    # 生成 UUID，并取最后16位
    uid = uuid.uuid4().hex[-16:]
    
    # 拼接 DOI
    doi = f"{timestamp}-{uid}"
    return doi

# 测试
if __name__ == "__main__":
    print(generate_doi())
