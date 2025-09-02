from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

import torch
import tqdm
from llm import *
from doi_utils import *
from graph import MemorySubgraph, Vectorizer
import numpy as np
import queue
from neo4j import GraphDatabase
from neo4j_utils import Neo4jTool,ENTITY_TYPE, CHUNK_TYPE, DOC_TYPE, RELATED_TO, BELONG_TO
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModel
from langchain_core.prompts import ChatPromptTemplate
from IPython.display import display, Markdown
from utils import safe_json_loads

def process_text_to_triplets(text_doi: str, text: str, queue: queue.Queue) -> str:
    try:
        raw_ontology = llm.invoke(make_graph_prompt.format_messages(context=text)).content
        print(f"[Info] process_text_to_raw_ontology: {raw_ontology}")
        ontology_text = llm.invoke(ontology_prompt.format_messages(context=text, ontology=raw_ontology)).content
        print(f"[Info] process_text_to_ontology_text: {ontology_text}")
        formated_text = llm.invoke(format_prompt.format_messages(context=ontology_text)).content
        print(f"[Info] process_text_to_formated_text: {formated_text}")
        triplets = json.loads(formated_text)
        for triplet in triplets:
            data = {"text_doi": text_doi, "triplet": triplet}
            queue.put(data)
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
            triplet["node_1"], ENTITY_TYPE,
            triplet["edge"], RELATED_TO,
            triplet["node_2"], ENTITY_TYPE
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

            doc_doi = generate_doi()
            # 添加文档节点
            subgraph.add_node(
                node_type=DOC_TYPE,
                name=os.path.basename(md_file),
                abstract=content[:512],
                doi=doc_doi,
                data_type="markdown"
            )

            # 使用文本分割器将长文本分割成多个小块
            splitter = RecursiveCharacterTextSplitter(
                #chunk_size=5000, #1500,
                chunk_size=2500, #1500,
                chunk_overlap=0,
                length_function=len,
                is_separator_regex=False,
            )
            texts = splitter.split_text(content)
            textidx_doi_map = {idx: generate_doi() for idx, text in enumerate(texts)}
            #添加chunk节点和BELONG_TO关系
            for idx, text in enumerate(texts):              
                text_doi = textidx_doi_map[idx]
                subgraph.add_node(
                    node_type=CHUNK_TYPE,
                    name=f"Chunk {idx} of {os.path.basename(md_file)}",
                    abstract=text[:512],
                    doi=text_doi,
                    data_type="text_chunk"
                )
                subgraph.add_edge(
                    e1=text_doi,
                    rel=BELONG_TO,
                    e2=doc_doi,
                    rel_type=BELONG_TO
                )
            # 使用线程池并发处理每个文本块
            # 使用队列来收集结果
            q = queue.Queue()
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(process_text_to_triplets,textidx_doi_map[idx], text, q) for idx,text in enumerate(texts)]

                # 等待所有任务完成（可选，确保异常能抛出）
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"任务出错: {e}")
            # process_text_to_triplets(texts[0], q)
            # 从队列中获取所有三元组并添加到内存子图
            while not q.empty():
                data = q.get()
                text_doi = data["text_doi"]
                triplet = data["triplet"]
                #添加节点间的关系
                subgraph.add_triplet(
                    triplet["node_1"], ENTITY_TYPE,
                    triplet["edge"], RELATED_TO,
                    triplet["node_2"], ENTITY_TYPE
                )
                #添加节点和chunk的关系
                subgraph.add_edge(
                    e1=triplet["node_1"],
                    rel=BELONG_TO,
                    e2=text_doi,
                    rel_type=BELONG_TO
                )
                subgraph.add_edge(
                    e1=triplet["node_2"],
                    rel=BELONG_TO,
                    e2=text_doi,
                    rel_type=BELONG_TO
                )
            # 将子图插入到 Neo4j

            neo4jTool.insert_subgraph(subgraph)
            # 在日志中记录已经处理的文件名
            with open("processed_files.log", "a") as log_file:
                log_file.write(f"{md_file}\n")

    finally:
        end_time = time.time()
        print(f"Graph built in {end_time - start_time:.2f} seconds")

def json_to_formatted_text(json_data):
    formatted_text = ""

    formatted_text += f"### 核心假设\n{json_data['hypothesis']}\n\n"
    formatted_text += f"### 可能发现\n{json_data['outcome']}\n\n"
    formatted_text += f"### 机制\n{json_data['mechanisms']}\n\n"

    formatted_text += "### 设计原则\n"

    design_principles_list=json_data['design_principles']

    if isinstance(design_principles_list, list):
        for principle in design_principles_list:
            formatted_text += f"- {principle}\n"
    else:
        formatted_text += f"- {design_principles_list}\n"

    formatted_text += "\n"

    formatted_text += f"### 额外特性\n{json_data['unexpected_properties']}\n\n"
    formatted_text += f"### 对比\n{json_data['comparison']}\n\n"
    formatted_text += f"### 创新性\n{json_data['novelty']}\n"

    return formatted_text

def sci_agent(save_path: str):
    keyword1 = "quantum computing"
    keyword2 = "machine learning"
    # 初始化向量生成器
    tokenizer_model = "BAAI/bge-large-en-v1.5"
    vec_model = AutoModel.from_pretrained(tokenizer_model).to("cuda")
    vec_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    vec_gen = Vectorizer(model=vec_model, tokenizer=vec_tokenizer, backend="numpy", use_gpu=True)
    # 初始化子图
    neo4jTool = Neo4jTool(uri="bolt://localhost:7687", user="neo4j", password="test1234", threshold=0.97, vectorizer=vec_gen)
    graph_path = neo4jTool.query_path(keyword1, keyword2)

    ontologist_result = ontologist.invoke(ontologist.format_messages(
        first_keyword=keyword1,
        last_keyword=keyword2,
        path_str=" -- ".join(graph_path))).content
    
    idea_generater_result = idea_generater.invoke(idea_generater.format_messages(
        first_keyword=keyword1,
        last_keyword=keyword2,
        path_str=" -- ".join(graph_path),
        ontologist_result=ontologist_result)).content
    
    idea_dict = safe_json_loads(idea_generater_result)
    formatted_text = json_to_formatted_text(idea_dict)
    idea_expanded_dict = {}
    expanded_text = ''
    for i, field in tqdm(enumerate (list (idea_dict.keys())[:7])):
        print(f"[Info] Expanding field {i}: {field}")
        field_content = idea_dict[field]
        expanded_content = llm.invoke(idea_expander.format_messages(
            idea=formatted_text,
            first_keyword=keyword1,
            last_keyword=keyword2,
            path_str=" -- ".join(graph_path),
            idea_field=field,
            idea_content=field_content)).content
        idea_expanded_dict[field] = expanded_content
        expanded_text = expanded_text+f'\n\n'+expanded_content
    complete = (
        f"# '{keyword1}' 和 '{keyword2}'之间的概念研究\n\n"
        f"### 知识图谱:\n\n{' -- '.join(graph_path)}\n\n"
        f"### 拓展的图谱:\n\n{ontologist_result}"
        f"### 提议的研究 / 材料:\n\n{formatted_text}"
        f"\n\n### 拓展描述:\n\n{expanded_text}"
    )
    critiques = critist.invoke(critist.format_messages(doc_text=complete)).content
    complete_doc=complete+ f'\n\n## 摘要、批判性评审与改进建议:\n\n'+critiques
    advice = adviser.invoke(adviser.format_messages(complete_doc=complete_doc)).content
    complete_doc=complete_doc+ f'\n\n## 建模与模拟的重点方向:\n\n'+advice
    #按时间给生成文件命名
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = save_path+f"/sci_agent_output_{timestamp}.md"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(complete_doc)
    display (Markdown(complete_doc[:256]+"...."))




if __name__ == "__main__":
    build_graph()
    