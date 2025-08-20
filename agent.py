from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from llm import *
from graph import MemorySubgraph, Vectorizer
import numpy as np
import queue
from neo4j import GraphDatabase
from neo4j_utils import Neo4jTool,CONCEPT_TYPE, CORPUS_TYPE, DOC_TYPE, RELATED_TO, BELONG_TO
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModel
from langchain_core.prompts import ChatPromptTemplate
from IPython.display import display, Markdown

def process_text_to_triplets(text: str,queue: queue.Queue) -> str:
    try:
        raw_ontology = llm.invoke(make_graph_prompt.format_messages(context=text)).content
        print(f"[Info] process_text_to_raw_ontology: {raw_ontology}")
        ontology_text = llm.invoke(ontology_prompt.format_messages(context=text, ontology=raw_ontology)).content
        print(f"[Info] process_text_to_ontology_text: {ontology_text}")
        formated_text = llm.invoke(format_prompt.format_messages(context=ontology_text)).content
        print(f"[Info] process_text_to_formated_text: {formated_text}")
        triplets = json.loads(formated_text)
        for triplet in triplets:
            queue.put(triplet)
    except Exception as e:
        print(f"[Error] process_text_to_triplets failed: {e}")

def make_graph_from_text(text: str, subgraph: MemorySubgraph) -> MemorySubgraph:
    """
    从文本中提取概念和关系，生成图谱
    :param text: 输入文本
    :return: MemorySubgraph 对象
    """
    # 使用 LLM 生成图谱三元组
    response = llm.invoke(make_graph_prompt.format_messages(context=text))
    
    # 解析 LLM 响应
    triplets = eval(response.content)  # 假设返回的格式是 JSON 字符串
    
    # 添加三元组到子图
    for triplet in triplets:
        subgraph.add_triplet(
            triplet["node_1"], CONCEPT_TYPE,
            triplet["edge"], RELATED_TO,
            triplet["node_2"], CONCEPT_TYPE
        )
    
    return subgraph

# 获取指定文件夹下的所有 Markdown 文件
def get_all_md_files(folder_path):
    md_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.md'):
                md_files.append(os.path.join(root, file))
    return md_files

# 构建图谱
def build_graph():
    md_file_folder = '/home/liang/SciAgentsDiscovery/papers/Al/test'
    md_files = get_all_md_files(md_file_folder)
    start_time = time.time()
    # 初始化向量生成器
    tokenizer_model = "BAAI/bge-large-en-v1.5"
    vec_model = AutoModel.from_pretrained(tokenizer_model).to("cuda")
    vec_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    vec_gen = Vectorizer(model=vec_model, tokenizer=vec_tokenizer, backend="numpy", use_gpu=True)
    # 初始化子图
    subgraph = MemorySubgraph(similarity_threshold=0.97, use_vector=True, vectorizer=vec_gen)
    neo4jTool = Neo4jTool(uri="bolt://localhost:7687", user="neo4j", password="test1234", threshold=0.97, vectorizer=vec_gen)

    try:
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            display (Markdown(content[:256]+"...."))

            # 使用文本分割器将长文本分割成多个小块
            splitter = RecursiveCharacterTextSplitter(
                #chunk_size=5000, #1500,
                chunk_size=2500, #1500,
                chunk_overlap=0,
                length_function=len,
                is_separator_regex=False,
            )
            texts = splitter.split_text(content)

            # 使用线程池并发处理每个文本块
            # 使用队列来收集结果
            q = queue.Queue()
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(process_text_to_triplets, text, q) for text in texts]

                # 等待所有任务完成（可选，确保异常能抛出）
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"任务出错: {e}")
            # process_text_to_triplets(texts[0], q)
            # 从队列中获取所有三元组并添加到内存子图
            while not q.empty():
                triplet = q.get()
                subgraph.add_triplet(
                    triplet["node_1"], CONCEPT_TYPE,
                    triplet["edge"], RELATED_TO,
                    triplet["node_2"], CONCEPT_TYPE
                )
            # 将子图插入到 Neo4j

            neo4jTool.insert_subgraph(subgraph)
            # 在日志中记录已经处理的文件名
            with open("processed_files.log", "a") as log_file:
                log_file.write(f"{md_file}\n")

    finally:
        end_time = time.time()
        print(f"Graph built in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    build_graph()
    