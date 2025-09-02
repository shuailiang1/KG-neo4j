from neo4j import GraphDatabase
from typing import List, Optional
import numpy as np
from graph import MemorySubgraph, Vectorizer
import torch

# Neo4j 节点类型常量
ENTITY_TYPE = "Entity"  # Neo4j 中的概念节点类型
CHUNK_TYPE = "Chunk"  # Neo4j 中的语料库节点类型
DOC_TYPE = "Document"  # Neo4j 中的文档节点类型
# Neo4j 边类型常量
RELATED_TO = "RELATED_TO"  # Neo4j 中的关系类型
BELONG_TO = "BELONG_TO"  # Neo4j 中的归属关系类型

class Neo4jTool:
    def __init__(self, uri: str, user: str, password: str, threshold: float = 0.9, vectorizer: Optional[Vectorizer] = None):
        """
        Neo4j 图谱写入工具
        :param uri: Neo4j bolt URI
        :param user: 用户名
        :param password: 密码
        :param threshold: 节点向量相似度阈值
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.threshold = threshold
        self.vectorizer = vectorizer
        self._ensure_vector_index()

    def close(self):
        self.driver.close()

    def _ensure_vector_index(self):
        """确保 Entity 节点的向量索引存在"""
        query = """
        CREATE VECTOR INDEX concept_embedding_index
        IF NOT EXISTS
        FOR (c:Entity) ON (c.embedding)
        OPTIONS { indexConfig: {
            `vector.dimensions`: 1024,
            `vector.similarity_function`: 'cosine'
        }}
        """
        with self.driver.session() as session:
            session.run(query)

    def _find_similar_node(self, embedding: List[float]) -> Optional[int]:
        """
        使用向量索引查找是否已有相似节点
        :param embedding: 节点向量
        :return: 节点 id（int）或 None
        """
        with self.driver.session() as session:
            result = session.run("""
            CALL db.index.vector.queryNodes('entity_embedding_index', 1, $embedding)
            YIELD node, score
            WHERE score >= $threshold
            RETURN id(node) AS node_id, score
            """, embedding=embedding, threshold=self.threshold)
            record = result.single()
            return record["node_id"] if record else None
    def find_node_with_embedding(self, embedding: List[float]) -> Optional[int]:
        """
        使用向量索引查找是否已有相似节点
        :param embedding: 节点向量
        :return: 节点 id（int）或 None
        """
        with self.driver.session() as session:
            result = session.run("""
            CALL db.index.vector.queryNodes('entity_embedding_index', 1, $embedding)
            YIELD node, score
            WHERE score >= $threshold
            RETURN node
            """, embedding=embedding, threshold=self.threshold)
            record = result.single()
            return record["node"]["name"] if record else None
    def find_node_with_name(self, name: str) -> Optional[int]:
        """
        使用向量索引查找是否已有相似节点
        :param name: 节点名称
        :return: 节点 name 或 None
        """
        embedding = self._get_embedding(name)
        with self.driver.session() as session:
            result = session.run("""
            CALL db.index.vector.queryNodes('entity_embedding_index', 1, $embedding)
            YIELD node, score
            WHERE score >= $threshold
            RETURN node
            """, embedding=embedding, threshold=self.threshold)
            record = result.single()
            return record["node"]["name"] if record else None
        
    def _sanitize_rel_type(self, rel: str) -> str:
        """
        把任意字符串转成合法的 Neo4j 关系类型
        - 空格 -> "_"
        - 非字母数字下划线的字符 -> "_"
        - 必须以字母开头，不是字母的话加前缀 REL_
        """
        import re
        rel_clean = re.sub(r"[^0-9A-Za-z_]", "_", rel)
        if not rel_clean[0].isalpha():
            rel_clean = "REL_" + rel_clean
        return rel_clean.upper()  # 关系类型一般习惯用大写
    def insert_subgraph(self, memory_subgraph: MemorySubgraph):
        """
        将内存子图存入 Neo4j（自动去重，基于向量相似度）
        :param memory_subgraph: MemorySubgraph 实例
        """
        with self.driver.session() as session:
            node_id_map = {}  # MemorySubgraph id -> Neo4j node id

            # 处理节点，已做去重处理
            for node in memory_subgraph.get_nodes():
                mem_id = node.get("id")
                node_type = node.get("node_type", "Entity")   # 默认为 Entity
                embedding = node.get("embedding")

                # 整理节点属性（统一用 props 存储）
                props = {k: v for k, v in node.items() if k not in ["id", "node_type"]}

                neo4j_id = None
                if embedding is not None:
                    neo4j_id = self._find_similar_node(embedding)

                if neo4j_id:  
                    # 已存在，复用
                    node_id_map[mem_id] = neo4j_id
                else:  
                    # 插入新节点，直接用 props
                    result = session.run(f"""
                    CREATE (c:{node_type})
                    SET c += $props
                    RETURN id(c) AS node_id
                    """, props=props)
                    node_id_map[mem_id] = result.single()["node_id"]

            # 处理边
            for edge in memory_subgraph.get_edges():
                src = node_id_map[edge["source"]]
                tgt = node_id_map[edge["target"]]

                # 统一关系类型
                rel_type = edge.get("rel_type", "RELATED_TO")

                # 保留边的其他属性
                props = {k: v for k, v in edge.items() if k not in ["source", "target", "rel_type"]}
                props["relation"] = edge.get("relation", "RELATED_TO")

                session.run(f"""
                MATCH (a),(b)
                WHERE id(a) = $src AND id(b) = $tgt
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += $props
                """, src=src, tgt=tgt, props=props)


    
    def _get_embedding(self, text: str) -> List[float]:
        return self.vectorizer.encode(text).tolist() 

    def query_path(self, start_text, end_text, mode="random", max_depth=64) -> List[str]:
        """
        查询路径（最短路径 or 随机路径）
        返回路径的节点与边列表
        """
        start_emb = self._get_embedding(start_text)
        end_emb = self._get_embedding(end_text)

        # 找到最相似的起点和终点节点
        start_node = self.driver.execute_query(
            "CALL db.index.vector.queryNodes('entity_embedding_index', 1, $embedding) YIELD node, score RETURN node",
            {"embedding": start_emb}
        ).records[0]["node"] 
        end_node = self.driver.execute_query(
            "CALL db.index.vector.queryNodes('entity_embedding_index', 1, $embedding) YIELD node, score RETURN node",
            {"embedding": end_emb}
        ).records[0]["node"] 

        if not start_node or not end_node:
            return []

        start_name = start_node["name"]
        end_name = end_node["name"]

        with self.driver.session() as session:
            if mode == "shortest":
                query = f"""
                    MATCH p=shortestPath((a:Entity {{name:$start}})-[*..{max_depth}]-(b:Entity {{name:$end}}))
                    RETURN p
                """
            elif mode == "random":
                query = f"""
                MATCH p=(a:Entity {{name:$start}})-[*..{max_depth}]-(b:Entity {{name:$end}})
                WITH p, rand() AS r
                RETURN p ORDER BY r LIMIT 1
                """
            else:
                raise ValueError("mode 必须是 'shortest' 或 'random'")
            
            result = session.run(query, start=start_name, end=end_name)
            paths = []
            for record in result:
                path = record["p"]
                path_list = []
                nodes = path.nodes
                rels = path.relationships

                for i in range(len(nodes)):
                    # 先加入节点 name
                    path_list.append(nodes[i]["name"])
                    # 如果有对应的边，加入边的内容（可以选择边的 type 或属性）
                    if i < len(rels):
                        # 这里我把边的 type 和所有属性都放成字典
                        edge_info = rels[i]["relation"]
                        path_list.append(edge_info)

                paths.append(path_list)
            return paths if paths else None
