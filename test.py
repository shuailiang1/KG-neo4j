from graph import MemorySubgraph, Vectorizer
import numpy as np
from neo4j import GraphDatabase
from neo4j_utils import Neo4jTool,CONCEPT_TYPE, CORPUS_TYPE, DOC_TYPE, RELATED_TO, BELONG_TO
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModel
from llm import *

def test1():
    tokenizer_model = "BAAI/bge-large-en-v1.5"
    vec_model = AutoModel.from_pretrained(tokenizer_model).to("cuda")  # 假设有一个预训练模型
    vec_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)  # 假设有一个预训练模型的 tokenizer
    # 初始化向量生成器
    vec_gen = Vectorizer(model=vec_model,tokenizer=vec_tokenizer, backend="numpy", use_gpu=True)

    # 初始化子图
    subgraph = MemorySubgraph(similarity_threshold=0.9,
                            use_vector=True,
                            vectorizer=vec_gen)

    # 添加三元组（不传 vec，会自动生成）
    subgraph.add_triplet("cow",CONCEPT_TYPE, "produce",RELATED_TO , "milk", CONCEPT_TYPE)
    subgraph.add_triplet("milk_",CONCEPT_TYPE, "can be",RELATED_TO , "drink", CONCEPT_TYPE)  # 如果B2与B很相似，会合并
 
    print("Nodes:", subgraph.get_nodes())
    print("Edges:", subgraph.get_edges())
    neo4jTool = Neo4jTool(uri="bolt://localhost:7687", user="neo4j", password="test1234",threshold=0.98,vectorizer=vec_gen)
    neo4jTool.insert_subgraph(subgraph)
    # name = neo4jTool.find_node_with_name("cow")
    # print("Node ID for 'cow':", name)
    path = neo4jTool.query_path("cow", "drink", mode="shortest")
    print("Shortest path from 'cow' to 'drink':", path[0])
    path = neo4jTool.query_path("cow", "drink", mode="random")
    print("Random path from 'cow' to 'drink':", path[0])
    # 关闭 Neo4j 连接
    neo4jTool.close()

if __name__ == "__main__":
    print("{{}}")
