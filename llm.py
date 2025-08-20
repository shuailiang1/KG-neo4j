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