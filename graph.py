import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModel

try:
    import torch
except ImportError:
    torch = None


# embedding 生成器
class Vectorizer:
    def __init__(self, model: Optional[Any] = None,
                    tokenizer: Optional[Any] = None,
                 dim: int = 8,
                 use_gpu: bool = False,
                 backend: str = "numpy",
                 device: str = "cuda" ):
        """
        向量生成器
        :param model: 模型对象（可选，若为None则随机生成向量）
        :param dim: 向量维度
        :param use_gpu: 是否使用GPU (仅在backend='torch'时生效)
        :param backend: 'numpy' 或 'torch'
        """
        self.model = model
        self.tokenizer = tokenizer # 假设有一个 tokenizer，如果需要的话
        self.dim = dim
        self.use_gpu = use_gpu and torch is not None
        self.backend = backend
        self.device = device
        

    def encode(self, text: str) -> Union[np.ndarray, "torch.Tensor"]:
        """
        将文本转换为向量
        如果未提供模型，则返回随机向量（用于测试）
        """
        if self.model:
            # 假设模型有 encode 方法，返回 numpy 或 torch
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            vec = self.model(**inputs)
            return vec.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy()
        else:
            # 生成随机向量
            if self.backend == "torch" and torch is not None:
                device = "cuda" if self.use_gpu else "cpu"
                vec = torch.randn(self.dim, device=device)
            else:
                vec = np.random.randn(self.dim).astype(np.float32)

        return vec / (np.linalg.norm(vec) + 1e-8)  # 归一化

# 内存子图
class MemorySubgraph:
    def __init__(self,
                 similarity_threshold: float = 0.9,
                 use_vector: bool = True,
                 vectorizer: Optional[Vectorizer] = None):
        """
        内存子图
        :param similarity_threshold: 余弦相似度阈值
        :param use_vector: 是否使用向量
        :param vectorizer: 向量生成器
        """
        self.graph = nx.MultiDiGraph()
        self.similarity_threshold = similarity_threshold
        self.use_vector = use_vector
        self.vectorizer = vectorizer

    def _cosine_similarity(self, v1, v2) -> float:
        """支持 numpy 和 torch 的余弦相似度"""
        if torch is not None and isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            return float(torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item())
        else:
            v1, v2 = np.array(v1), np.array(v2)
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                return 0.0
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def _find_similar_node(self, vector) -> Optional[str]:
        """在已有节点中查找相似节点"""
        best_node, best_score = None, -1.0
        for node, data in self.graph.nodes(data=True):
            if "vector" in data and data["vector"] is not None:
                score = self._cosine_similarity(vector, data["vector"])
                if score > self.similarity_threshold and score > best_score:
                    best_score, best_node = score, node
        return best_node

    def add_triplet(self, e1: str, e1_type: str, rel: str,rel_type: str , e2: str, e2_type: str,
                    vec1: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
                    vec2: Optional[Union[np.ndarray, "torch.Tensor"]] = None):
        """
        添加三元组 (e1 -rel-> e2)
        如果 use_vector=True 且未传 vec，会调用 vectorizer.encode(e) 生成
        """
        if self.use_vector:
            if vec1 is None and self.vectorizer:
                vec1 = self.vectorizer.encode(e1)
            if vec2 is None and self.vectorizer:
                vec2 = self.vectorizer.encode(e2)

        # 检查 e1 是否已有相似节点
        e1_id = self._find_similar_node(vec1) if (self.use_vector and vec1 is not None) else None
        if not e1_id:
            e1_id = e1
            self.graph.add_node(e1_id, name=e1, type=e1_type, vector=vec1 if self.use_vector else None)

        # 检查 e2 是否已有相似节点
        e2_id = self._find_similar_node(vec2) if (self.use_vector and vec2 is not None) else None
        if not e2_id:
            e2_id = e2
            self.graph.add_node(e2_id, name=e2, type=e2_type, vector=vec2 if self.use_vector else None)

        # 添加关系
        self.graph.add_edge(e1_id, e2_id, relation=rel, type=rel_type)

    def get_nodes(self) -> List[Dict[str, Any]]:
        return [{**{"id": n}, **d} for n, d in self.graph.nodes(data=True)]

    def get_edges(self) -> List[Dict[str, Any]]:
        return [{"source": u, "target": v, **d} for u, v, d in self.graph.edges(data=True)]

    def __repr__(self):
        return f"<MemorySubgraph | Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}>"

if __name__ == "__main__":
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
    subgraph.add_triplet("cow", "produce", "milk")
    subgraph.add_triplet("milk_", "can be", "drink")  # 如果B2与B很相似，会合并

    print(subgraph)
    print("Nodes:", subgraph.get_nodes())
    print("Edges:", subgraph.get_edges())

