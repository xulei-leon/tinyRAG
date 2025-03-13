import os
import logging
import operator
from typing import List, Literal, Annotated, Optional, Union, Any
from typing_extensions import TypedDict
import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field, validator

from langchain import hub
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough, RunnableSerializable
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_chroma import Chroma
from langchain_text_splitters import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader
from langchain.schema import Document
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver


#
# key
#
from dotenv import load_dotenv

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
silicon_api_key = os.getenv("SILICON_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
linkup_api_key = os.getenv("LINKUP_API_KEY")

#
# config
#
import tomllib


def load_config(config_file):
    try:
        with open(config_file, "rb") as f:
            config = tomllib.load(f)
            return config
    except Exception as e:
        print(f"Load config file error: {e}")
        return None


# load config file
deepseek_llm_model = None
silicon_base_url = None
silicon_llm_model = None
huggingface_embed_model = None

try:
    config_data = load_config("config.toml")
    log_level = config_data.get("log_level")
    if log_level:
        logging.basicConfig(level=log_level)

    # deepseek
    deepseek_llm_model = config_data.get("deepseek", {}).get("model")
    deepseek_llm_temperature = config_data.get("deepseek", {}).get("temperature")
    deepseek_llm_max_tokens = config_data.get("deepseek", {}).get("max_tokens")

    # silicon
    silicon_base_url = config_data.get("silicon", {}).get("base_url")
    silicon_llm_model = config_data.get("silicon", {}).get("model")

    # huggingface
    huggingface_embed_model = config_data.get("huggingface", {}).get("embed_model")
except:
    # deepseek
    deepseek_llm_model = deepseek_llm_model or "deepseek-chat"
    deepseek_llm_temperature = 0.5
    deepseek_llm_max_tokens = None

    # silicon
    silicon_base_url = silicon_base_url or "https://api.siliconflow.cn/v1"
    silicon_llm_model = silicon_llm_model or "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    # huggingface
    huggingface_embed_model="sentence-transformers/all-MiniLM-L6-v2"


# init LLM mod
llm_deepseek = ChatDeepSeek(
    model=deepseek_llm_model,
    temperature=deepseek_llm_temperature,
    max_tokens=deepseek_llm_max_tokens,
    timeout=None,
    top_p=0.9,
    frequency_penalty=0.7,
    presence_penalty=0.5,
    max_retries=3,
)

################################################################################
### RAG state
################################################################################
class RagState(TypedDict):
    question: str
    answer: str
    retrieves: List[Document]
    retrieves_relevant: List[Document]
    retrieves_weak: List[Document]
    web_searchs: List[Document]
    #documents: Annotated[list, operator.add]


################################################################################
### RAG nodes
################################################################################
def node_rewrite_qestion(state: RagState) -> RagState:
    question = state["question"]
    better_question = None

    # Re-write question
    # better_question = question_rewriter.invoke({"question": question})
    if not better_question:
        better_question = question

    updated_state = state.copy()
    updated_state.update({"question": better_question})
    return updated_state

def node_retrieve(state: RagState) -> RagState:
    question = state["question"]
    retrieves = []

    # Retrieval
    # retrieves = retriever.invoke(question)
    
    updated_state = state.copy()
    updated_state.update({"retrieves": retrieves})
    return updated_state

def node_retrieve_grade(state: RagState) -> RagState:
    question = state["question"]
    retrieves = state["retrieves"]
    relevant_docs = []
    weak_docs = []

    # Score each doc
    for doc in retrieves:
        # grade = retrieval_grader.invoke(
        #    {"question": question, "document": doc.page_content}
        # )

        if grade:
            score = grade.score
        else:
            score = 0.5
            
        if score > 0.7:
            relevant_docs.append(doc)
        elif score >= 0.5:
            weak_docs.append(doc)

    updated_state = state.copy()
    updated_state.update({"retrieves_relevant": relevant_docs, "retrieves_weak": weak_docs})
    return updated_state

def node_web_search(state: RagState) -> RagState:
    question = state["question"]

    # Web search
    web_results = []
    # web_results = web_search_tool.invoke({"query": question})

    updated_state = state.copy()
    updated_state.update({"web_searchs": web_results})
    return updated_state


def node_generate(state: RagState) -> RagState:
    question = state["question"]
    retrieves_relevant = state.get("retrieves_relevant", None)
    retrieves_weak = state.get("retrieves_weak", None)
    web_searchs = state.get("web_searchs", None)

    # generation
    generation = "This is the answer!"

    updated_state = state.copy()
    updated_state.update({"answer":generation})
    return updated_state


################################################################################
### Edges conditional functions
################################################################################
def condition_retrieve(state: RagState) -> str:
    retrieves = state["retrieves"]

    if retrieves:
        return "success"
    else:
        return "failure"


def condition_retrieve_grade(state: RagState) -> str:
    retrieves_relevant = state["retrieves_relevant"] or None
    retrieves_weak = state["retrieves_weak"] or None

    if retrieves:
        return "Relevant"
    elif retrieves_weak:
        return "Weak"
    else:
        return "Irrelevant"
    
def rag_graph_init() -> StateGraph:
    ################################################################################
    ### Create Graph
    ################################################################################
    workflow = StateGraph(RagState)

    ################################################################################
    ### Add nodes
    ################################################################################
    workflow.add_node("rewrite_qestion", node_rewrite_qestion)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("retrieve_grade", node_retrieve_grade)
    workflow.add_node("web_search", node_web_search)
    workflow.add_node("generate", node_generate)

    ################################################################################
    ### Add edges
    ################################################################################
    ## rewrite qestion
    workflow.add_edge(START, "rewrite_qestion")
    workflow.add_edge("rewrite_qestion", "retrieve")

    ## retrieve
    workflow.add_conditional_edges(
        "retrieve",
        condition_retrieve,
        {
            "success": "retrieve_grade",
            "failure": "web_search",
        },
    )
    workflow.add_conditional_edges(
        "retrieve_grade",
        condition_retrieve_grade,
        {
            "Relevant": "generate",
            "Weak": "web_search",
            "Irrelevant": "web_search",
        },
    )

    ## web_search
    workflow.add_edge("web_search", "generate")

    ## generate
    workflow.add_edge("generate", END)

    return workflow

################################################################################
### Build RAG Graph
################################################################################
rag_graph = rag_graph_init()
rag_app = rag_graph.compile()

################################################################################
### Display Graph
### MUST install: pip install grandalf
################################################################################
#print(rag_app.get_graph().draw_ascii())
