import os
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
# 设置 API 密钥
api_key = os.environ.get("DASHSCOPE_API_KEY")

# Create the LLM
llm = ChatTongyi(model="qwen3-235b-a22b-instruct-2507",api_key=api_key)

# 从文本中提取概念和关系的prompt模板
make_graph_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a network ontology graph maker who extracts terms and their relations from a given context, using category theory. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context, including well-defined and widely used names of materials, systems, methods. \n\n"
        "Format your output as a list of JSON. Each element of the list contains a pair of terms"
        "and the relation between them, like the follwing: \n"
        "[\n"
        "   {{\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"\n'
        "   }}, {{...}}\n"
        "]"
        ""
        "Examples:"
        "Context: ```Alice is Marc's mother.```\n"
        "[\n"
        "   {{\n"
        '       "node_1": "Alice",\n'
        '       "node_2": "Marc",\n'
        '       "edge": "is mother of"\n'
        "   }}, "
        "{{...}}\n"
        "]"
        "Context: ```Silk is a strong natural fiber used to catch prey in a web. Beta-sheets control its strength.```\n"
        "[\n"
        "   {{\n"
        '       "node_1": "silk",\n'
        '       "node_2": "fiber",\n'
        '       "edge": "is"\n'
        "   }}," 
        "   {{\n"
        '       "node_1": "beta-sheets",\n'
        '       "node_2": "strength",\n'
        '       "edge": "control"\n'
        "   }},"        
        "   {{\n"
        '       "node_1": "silk",\n'
        '       "node_2": "prey",\n'
        '       "edge": "catches"\n'
        "   }},"
        "{{...}}\n"
        "]\n\n"
        "Analyze the text carefully and produce around 10 triplets, making sure they reflect consistent ontologies.\n"
        ),
        ("human", "Context: ```{context}``` \n\nOutput: "),
    ]
    )

ontology_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 'You respond in this format:'
                 '[\n'
                    "   {{\n"
                    '       "node_1": "A concept from extracted ontology",\n'
                    '       "node_2": "A related concept from extracted ontology",\n'
                    '       "edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"\n'
                    '   }}, {{...}} ]\n'  ),
        ("human", 'Read this context: ```{context}```.'
                  'Read this ontology: ```{ontology}```\n\n'
                 '\n\nImprove the ontology by renaming nodes so that they have consistent labels that are widely used in the field of materials science.'
                 ),
    ]
)
format_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 'You respond in this format:'
                 '[\n'
                    "   {{\n"
                    '       "node_1": "A concept from extracted ontology",\n'
                    '       "node_2": "A related concept from extracted ontology",\n'
                    '       "edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"\n'
                    '   }}, {{...}} ]\n'  ),
        ("human", "Context: ```{context}``` \n\n Fix to make sure it is proper format. "),
    ]
)


# 生成假设并验证的prompt模板
ontologist = ChatPromptTemplate.from_messages(
    [
        ("system", '你是一位富有创造力的科学家，能够提供准确、详尽且有价值的回答。'  ),
        ("human", """你是一位在科学研究、工程与创新方面受过训练的高级本体论专家。

        鉴于下列从一个综合知识图谱中提取的关键概念，你的任务是对每一个术语进行定义，并讨论图谱中所识别出的关系。

        请参考从知识图谱中提取的"{first_keyword}"与"{last_keyword}"之间的节点与关系清单。

        知识图谱的格式为：“节点1 —— 节点1与节点2之间的关系 —— 节点2 —— 节点2与节点3之间的关系 —— 节点3……”。

        以下是图谱路径：

        {path_str}

        请务必在你的回答中包含知识图谱中的每一个概念。

        不要添加任何引言性语句。首先，请定义知识图谱中的每个术语，然后逐一讨论每个关系，并结合语境说明。"""),
    ]
)

idea_generater = ChatPromptTemplate.from_messages(
    [
        ("system", '你是一位富有创造力的科学家，能够以 JSON 格式提供准确、详尽且有价值的回答。'  ),
        ("human", """你是一位在科学研究与创新方面受过训练的高级科学家。

鉴于下列从一个综合知识图谱中提取的关键概念，你的任务是综合提出一个全新的研究假设。你的回答不仅应体现深刻的理解与理性思维，还要探索这些概念的富有想象力和非传统的应用。

请参考从"{first_keyword}"与"{last_keyword}"之间的知识图谱中提取出的节点与关系清单。
图谱的格式为："节点1 —— 节点1与节点2之间的关系 —— 节点2 —— 节点2与节点3之间的关系 —— 节点3……"

以下是图谱路径：

{path_str}

以下是对图中概念和关系的分析：{ontologist_result}请深入且仔细地分析图谱，然后设计一个详细的研究假设，探索其中可能具有突破性的方面，并将图谱中的每一个概念都纳入考虑。请思考该假设的潜在影响，并预测由此研究方向可能产生的结果或行为。我们特别重视你在将这些概念联系起来解决未解问题、提出尚未探索的研究领域、预见新奇或意外行为等方面展现出的创造力。

请尽量使用定量数据，包括数字、序列或化学式等细节。请将你的回答用 JSON 格式 表达，包含以下七个键：

"hypothesis"：清晰陈述所提出的研究问题的核心假设。

"outcome"：描述该研究可能得出的发现或影响。请使用定量方式，包括数值、材料属性、序列、化学式等。

"mechanisms"：说明预期涉及的化学、生物或物理机制。请尽可能具体，涵盖从分子到宏观尺度的机制。

"design_principles"：列出详细的设计原则，聚焦新颖概念，信息应丰富并经过深思熟虑。

"unexpected_properties"：预测该新材料或系统的意外特性。请提供具体预测，并以逻辑和推理清晰解释原因。

"comparison"：提供与现有材料、技术或科学概念的详细对比。请详尽且尽量量化。

"novelty"：讨论该提案的创新之处，具体指出其相对于现有知识与技术的进步。

请确保你的科学假设既具有创新性，又具备逻辑合理性，能够拓展我们对所提供概念的理解或应用。

以下是你回答应使用的 JSON 结构示例：

{{
  "hypothesis": "...",
  "outcome": "...",
  "mechanisms": "...",
  "design_principles": "...",
  "unexpected_properties": "...",
  "comparison": "...",
  "novelty": "...",
}}

请记住，你的回答价值在于：科学发现、科学研究新路径的开辟以及潜在的技术突破，需具备细节充分、推理严密的特点。

请确保在你的回答中纳入知识图谱中的每一个概念。

请严格按照我给的json格式输出，键使用我给的英文，值使用中文，用英文双引号

"""),
    ]
)


idea_expander = ChatPromptTemplate.from_messages(
    [
        ("system", '你是一位富有创造力的科学家，能够提供准确、详尽且有价值的回答。'  ), 
        ("human", '''你获得了一个新的研究想法：

{idea}

该研究想法基于一个描述两个概念 {first_keyword} 和 {last_keyword} 之间关系的知识图谱：

{path_str}

现在，请仔细展开关于该特定领域：```{field}```。

请批判性地评估原始内容并加以改进。
尽可能添加更多细节和定量的科学信息，如化学式、数值、蛋白质序列、工艺条件、微观结构等。
包括明确的理论依据和逐步推理。
如果可能，请对具体的建模与仿真技术和代码、实验方法或特定分析进行评论。

请以同行评审者的视角，认真评估该初稿科学内容，任务是批判性审查并提升其科学性：

{idea_content}

不要添加任何引言语句，除公式、专业名词或其它不适合用中文回答的词汇外均用中文回答。你的回答应以标题开头：### 标题 ...

'''),
    ]
)


critist = ChatPromptTemplate.from_messages(
    [
        ("system", '你是一位严谨的科学家，能够提供准确、详尽且有价值的回答。'  ),
        ("human", """阅读以下文档:\n\n{doc_text}\n\n请提供 (1) a摘要：用一个段落总结本文档的内容，要求包含足够细节，如涉及的机制、相关技术、模型与实验、拟采用的方法等。 \
and (2) 科学性评审：进行一项全面且批判性的科学评估，包括优点与不足，并提出改进建议。请使用逻辑推理和科学方法加以支持。"""),
    ]
)

adviser = ChatPromptTemplate.from_messages(
    [
        ("system", '你是一位严谨的科学家，能够提供准确、详尽且有价值的回答。'  ),
        ("human", """阅读这个文档:\n\n{complete_doc}\n\n请从该文档中识别出能够通过分子建模解决的最具影响力的科学问题。 \
\n\n概述建立和进行此类建模与模拟的关键步骤，包含具体细节，并强调计划工作的独特之处。"""),
    ]
)